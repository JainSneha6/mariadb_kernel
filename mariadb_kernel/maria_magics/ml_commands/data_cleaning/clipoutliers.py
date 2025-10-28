# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.
from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np
from collections import namedtuple
import enum
from typing import Callable, List, NamedTuple, Tuple
import pandas
from pandas.core.frame import DataFrame
# note: we don't strictly rely on SqlFetch import path. We'll attempt to use it if available.
try:
    from mariadb_kernel.sql_fetch import SqlFetch  # optional; used if present
except Exception:
    SqlFetch = None
import logging
import math
from datetime import datetime
import re
import os


class ClipOutliers(MariaMagic):
    """
    %clipoutliers [columns=col1,col2,...] [method=iqr|zscore]
                  [k=1.5] [z_thresh=3.0] [inplace=True|False]
    Clamps (clips) extreme values to computed boundary limits.
    - method:
        iqr -> Tukey IQR method using k (default 1.5)
        zscore -> mean ± z_thresh * std (default z_thresh=3.0)
    - columns: comma-separated list of columns to operate on. If omitted, all numeric columns are used.
    - inplace: if True (default) modifies data["last_select"] in-place.
               if False stores clipped copy in data["last_select_clipped"].
    Examples:
      %clipoutliers -> clip numeric columns using iqr (k=1.5) in-place
      %clipoutliers method=zscore z_thresh=2.5 columns=age,salary inplace=False
    Additionally, execution metadata is stored into a table `magic_metadata`.
    """
    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "clipoutliers"

    def help(self):
        return (
            "%clipoutliers [columns=col1,col2,...] [method=iqr|zscore] "
            "[k=1.5] [z_thresh=3.0] [inplace=True|False]\n"
            "Clamps extreme numeric values to computed boundaries (in-place by default)."
            "Execution metadata is recorded in table `magic_metadata`."
        )

    def _str_to_obj(self, s):
        """Convert strings like numbers or bools into Python objects."""
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                pass
        try:
            return bool(util.strtobool(s))
        except Exception:
            if isinstance(s, str) and len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
                return s[1:-1]
            return s

    def parse_args(self, input_str):
        """Parse key=value arguments."""
        if not input_str or input_str.strip() == "":
            return {}
        pairs = dict(token.split("=", 1) for token in shlex.split(input_str))
        for k, v in pairs.items():
            pairs[k] = self._str_to_obj(v)
        return pairs

    def _send_html(self, kernel, df):
        """Display DataFrame as HTML."""
        try:
            html = df.to_html(index=False)
            mime = "text/html"
        except Exception:
            html = str(df)
            mime = "text/plain"
        kernel.send_response(kernel.iopub_socket, "display_data",
                             {"data": {mime: html}, "metadata": {}})

    def _compute_bounds(self, series, method, k=1.5, z_thresh=3.0):
        """Compute (lower, upper) clipping bounds."""
        s = series.dropna()
        if s.empty:
            return None, None
        if method == "iqr":
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            return lower, upper
        elif method == "zscore":
            mean = s.mean()
            std = s.std()
            if std == 0 or np.isnan(std):
                return None, None
            lower = mean - z_thresh * std
            upper = mean + z_thresh * std
            return lower, upper
        else:
            raise ValueError(f"Unknown method {method}")

    # ---- New DB / metadata helpers ----
    def _sql_escape(self, val):
        """Escape a value for SQL single-quoted literal insert. None -> NULL"""
        if val is None:
            return "NULL"
        if not isinstance(val, str):
            val = str(val)
        # double single-quotes for SQL escaping
        return "'" + val.replace("'", "''") + "'"

    def _get_mariadb_client(self, kernel):
        """Return mariadb_client if present on kernel, else None"""
        return getattr(kernel, "mariadb_client", None)

    def _get_logger(self, kernel):
        """Return a logger on kernel if present, else create a temporary logger"""
        return getattr(kernel, "log", logging.getLogger(__name__))

    def _get_db_name(self, kernel):
        """
        Attempt to determine the currently used DB.
        Prefer SqlFetch if available; otherwise run SELECT DATABASE(); and try to parse.
        Returns empty string if none found.
        """
        mariadb_client = self._get_mariadb_client(kernel)
        log = self._get_logger(kernel)

        # Try SqlFetch if available
        if SqlFetch is not None and mariadb_client is not None:
            try:
                sf = SqlFetch(mariadb_client, log)
                dbname = sf.get_db_name()
                if isinstance(dbname, str):
                    return dbname
            except Exception:
                # fallthrough to manual approach
                log.debug("SqlFetch available but .get_db_name() failed; falling back.")
        # Fallback: run SELECT DATABASE();
        if mariadb_client is None:
            return ""
        try:
            # mariadb_client.run_statement may return HTML or "Query OK". Use pandas to parse if HTML.
            result = mariadb_client.run_statement("SELECT DATABASE();")
            if mariadb_client.iserror():
                # can't get db name
                return ""
            if not result:
                return ""
            # If result is raw HTML table, try to parse with pandas
            try:
                df_list = pandas.read_html(result)
                if df_list and isinstance(df_list, list) and len(df_list) > 0:
                    val = df_list[0].iloc[0, 0]
                    if isinstance(val, float) and pandas.isna(val):
                        return ""
                    return str(val) if val is not None else ""
            except Exception:
                # if not parseable by pandas, try regex to extract first cell content
                m = re.search(r"<td.*?>(.*?)</td>", result, flags=re.S | re.I)
                if m:
                    txt = re.sub(r"<.*?>", "", m.group(1))  # strip tags
                    txt = txt.strip()
                    if txt.lower() == "null" or txt == "":
                        return ""
                    return txt
                # If result is plain text (like the DB name)
                txt = str(result).strip()
                if txt.lower() == "null" or txt == "":
                    return ""
                return txt
        except Exception:
            return ""
        return ""

    def _ensure_metadata_table(self, kernel, db_name):
        """
        Create magic_metadata table if it doesn't exist.
        Columns: id, command_name, arguments, execution_timestamp,
                 affected_columns, operation_status, message, db_name, user_name
        """
        mariadb_client = self._get_mariadb_client(kernel)
        log = self._get_logger(kernel)

        if mariadb_client is None:
            # nothing to do
            return

        # Use db-qualified name if db_name is present; otherwise create in current schema
        table_full_name = f"{db_name}.magic_metadata" if db_name else "magic_metadata"

        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_full_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            command_name VARCHAR(255),
            arguments TEXT,
            execution_timestamp DATETIME,
            affected_columns TEXT,
            operation_status VARCHAR(50),
            message TEXT,
            db_name VARCHAR(255),
            user_name VARCHAR(255)
        );
        """
        try:
            mariadb_client.run_statement(create_sql)
            if mariadb_client.iserror():
                log.error(f"Error creating magic_metadata table: {mariadb_client.run_statement('SHOW WARNINGS;')}")
        except Exception as e:
            log.error(f"Failed to ensure magic_metadata table: {e}")

    def _insert_metadata(self, kernel, command_name, arguments, affected_columns,
                         operation_status, message, db_name, user_name):
        """
        Insert a metadata row into magic_metadata. Uses NOW() for timestamp.
        """
        mariadb_client = self._get_mariadb_client(kernel)
        log = self._get_logger(kernel)
        if mariadb_client is None:
            return

        table_full_name = f"{db_name}.magic_metadata" if db_name else "magic_metadata"

        # Escape values
        args_sql = self._sql_escape(arguments)
        affected_sql = self._sql_escape(affected_columns)
        status_sql = self._sql_escape(operation_status)
        message_sql = self._sql_escape(message)
        db_sql = self._sql_escape(db_name)
        user_sql = self._sql_escape(user_name)

        insert_sql = f"""
        INSERT INTO {table_full_name}
            (command_name, arguments, execution_timestamp, affected_columns,
             operation_status, message, db_name, user_name)
        VALUES (
            {self._sql_escape(command_name)},
            {args_sql},
            NOW(),
            {affected_sql},
            {status_sql},
            {message_sql},
            {db_sql},
            {user_sql}
        );
        """
        try:
            mariadb_client.run_statement(insert_sql)
            # swallow errors but log
            if mariadb_client.iserror():
                log.error("Error inserting into magic_metadata: %s", insert_sql)
        except Exception as e:
            log.error(f"Exception while inserting metadata: {e}")

    def _get_user_name(self, kernel):
        """Try several places to find the current user name; fallback to OS login or empty string."""
        candidates = [
            getattr(kernel, "user_name", None),
            getattr(kernel, "username", None),
            getattr(kernel, "user", None),
            getattr(kernel, "session", None),
            # might be kernel.user.identity etc. Try simple introspection:
        ]
        for cand in candidates:
            # cand might be an object; try str if not None
            if cand is None:
                continue
            if isinstance(cand, str) and cand.strip():
                return cand
            try:
                # if session-like object with 'user' attribute
                maybe = getattr(cand, "user", None)
                if isinstance(maybe, str) and maybe.strip():
                    return maybe
            except Exception:
                pass
        try:
            return os.getlogin()
        except Exception:
            return ""

    # ---- End DB helpers ----

    def execute(self, kernel, data):
        """Execute the %clipoutliers magic with metadata logging."""
        df = data.get("last_select")
        if df is None:
            kernel._send_message("stderr", "No last_select found in kernel data.")
            return
        if hasattr(df, "empty") and df.empty:
            kernel._send_message("stderr", "There is no data to process (empty DataFrame).")
            return
        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return
        # parse args
        columns_arg = args.get("columns", None)
        if isinstance(columns_arg, str):
            columns = [c.strip() for c in columns_arg.split(",") if c.strip()]
        elif isinstance(columns_arg, (list, tuple)):
            columns = list(columns_arg)
        else:
            columns = None
        method = str(args.get("method", "iqr")).lower()
        if method not in {"iqr", "zscore"}:
            kernel._send_message("stderr", f"Unknown method '{method}'. Allowed: iqr, zscore.")
            return
        try:
            k = float(args.get("k", 1.5))
        except Exception:
            k = 1.5
        try:
            z_thresh = float(args.get("z_thresh", 3.0))
        except Exception:
            z_thresh = 3.0
        inplace = bool(args.get("inplace", True))
        # Determine numeric columns
        if columns is None:
            target_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        else:
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                kernel._send_message("stderr", f"Column(s) not found: {', '.join(missing_cols)}")
                return
            target_columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            non_numeric = [c for c in columns if c not in target_columns]
            if non_numeric:
                kernel._send_message("stdout", f"Warning: non-numeric columns skipped: {', '.join(non_numeric)}")
        if not target_columns:
            kernel._send_message("stderr", "No numeric target columns found to clip outliers.")
            return

        # Prepare metadata context
        db_name = self._get_db_name(kernel)
        user_name = self._get_user_name(kernel)
        # ensure metadata table exists
        try:
            self._ensure_metadata_table(kernel, db_name)
        except Exception:
            # log but continue
            try:
                kernel._send_message("stdout", "Warning: failed to ensure metadata table (continuing).")
            except Exception:
                pass

        target_df = df if inplace else df.copy(deep=True)
        messages = []
        total_clipped = 0
        operation_status = "success"
        try:
            for col in target_columns:
                try:
                    series = target_df[col]
                    lower, upper = self._compute_bounds(series, method, k=k, z_thresh=z_thresh)
                    if lower is None and upper is None:
                        messages.append(f"Column '{col}': insufficient data to compute bounds; skipped.")
                        continue
                    # find how many will change
                    mask = ((series < lower) | (series > upper)) & ~series.isna()
                    n_changed = int(mask.sum())
                    # clip
                    target_df[col] = series.clip(lower=lower, upper=upper)
                    total_clipped += n_changed
                    messages.append(f"Column '{col}': clipped {n_changed} value(s) (bounds: {lower:.4f}, {upper:.4f}).")
                except Exception as e:
                    messages.append(f"Column '{col}': error while clipping: {e}")
            # finish up
            if inplace:
                data["last_select"] = target_df
                location_msg = "Modified in-place: data['last_select'] updated."
            else:
                data["last_select_clipped"] = target_df
                location_msg = "Result stored in data['last_select_clipped'] (original unchanged)."
            kernel._send_message("stdout", f"Clip outliers completed using {method}.\n"
                                         + "\n".join(messages)
                                         + f"\nTotal values clipped: {total_clipped}. {location_msg}")
        except Exception as e:
            operation_status = "error"
            messages.append(f"Fatal error during clipping: {e}")
            kernel._send_message("stderr", f"Fatal error during clipping: {e}")

        # Attempt to insert metadata (best-effort)
        try:
            args_for_db = self.args if isinstance(self.args, str) else str(self.args)
            affected_columns_str = "\n".join(target_columns)
            message_str = "\n".join(messages)
            self._insert_metadata(
                kernel=kernel,
                command_name=self.name(),
                arguments=args_for_db,
                affected_columns=affected_columns_str,
                operation_status=operation_status,
                message=message_str,
                db_name=db_name,
                user_name=user_name
            )
        except Exception as e:
            # metadata failure shouldn't interrupt user, but warn
            try:
                kernel._send_message("stdout", f"Warning: failed to write metadata: {e}")
            except Exception:
                pass

        # Show output (DataFrame)
        try:
            self._send_html(kernel, target_df)
        except Exception:
            pass
