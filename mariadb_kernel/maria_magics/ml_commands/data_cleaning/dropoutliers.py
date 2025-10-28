# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np
from collections import namedtuple
import logging
import os
import re

# Optional helper to reliably get current DB name (if available in environment)
try:
    from mariadb_kernel.sql_fetch import SqlFetch
except Exception:
    SqlFetch = None


class DropOutliers(MariaMagic):
    """
    %dropoutliers [columns=col1,col2,...] [method=iqr|zscore] [k=1.5] [z_thresh=3.0]

    Removes rows (IN-PLACE) from data['last_select'] where any selected numeric column
    is detected as an outlier according to the chosen method.

    Additionally logs execution metadata into `magic_metadata` table:
      id, command_name, arguments, execution_timestamp, affected_columns,
      operation_status, message, db_name, user_name
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "dropoutliers"

    def help(self):
        return (
            "%dropoutliers [columns=col1,col2,...] [method=iqr|zscore] [k=1.5] [z_thresh=3.0]\n"
            "Removes rows containing outliers from data['last_select'] (in-place).\n"
            "Execution metadata is recorded in table `magic_metadata`."
        )

    def _str_to_obj(self, s):
        """Cast simple strings to Python objects where sensible."""
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
        """Parse key=value arguments (keeps behavior consistent with other magics)."""
        if not input_str or input_str.strip() == "":
            return {}
        pairs = dict(token.split("=", 1) for token in shlex.split(input_str))
        for k, v in pairs.items():
            pairs[k] = self._str_to_obj(v)
        return pairs

    def _send_html(self, kernel, df):
        """Display DataFrame as HTML (fallback to text if needed)."""
        try:
            html = df.to_html(index=False)
            mime = "text/html"
        except Exception:
            html = str(df)
            mime = "text/plain"
        display_content = {"data": {mime: html}, "metadata": {}}
        kernel.send_response(kernel.iopub_socket, "display_data", display_content)

    def _detect_outliers_series(self, series, method, k=1.5, z_thresh=3.0):
        """Return boolean mask of outliers for a pandas Series (True where outlier)."""
        if series.dropna().empty:
            return pd.Series(False, index=series.index)

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            mask = (series < lower) | (series > upper)
            return mask.fillna(False)

        elif method == "zscore":
            mean = series.mean(skipna=True)
            std = series.std(skipna=True)
            if std == 0 or np.isnan(std):
                return pd.Series(False, index=series.index)
            z = (series - mean) / std
            mask = z.abs() > float(z_thresh)
            return mask.fillna(False)

        else:
            raise ValueError(f"Unknown method {method}")

    # --- metadata / DB helper methods (best-effort) ---
    def _get_mariadb_client(self, kernel):
        return getattr(kernel, "mariadb_client", None)

    def _get_logger(self, kernel):
        return getattr(kernel, "log", logging.getLogger(__name__))

    def _sql_escape(self, val):
        """Escape value to single-quoted SQL literal (None -> NULL)."""
        if val is None:
            return "NULL"
        if not isinstance(val, str):
            val = str(val)
        return "'" + val.replace("'", "''") + "'"

    def _get_db_name(self, kernel):
        """
        Attempt to determine current DB. Use SqlFetch if present; otherwise run SELECT DATABASE(); parse result.
        Returns empty string if none.
        """
        mariadb_client = self._get_mariadb_client(kernel)
        log = self._get_logger(kernel)

        if SqlFetch is not None and mariadb_client is not None:
            try:
                sf = SqlFetch(mariadb_client, log)
                return sf.get_db_name() or ""
            except Exception:
                log.debug("SqlFetch.get_db_name() failed; falling back to manual query.")

        if mariadb_client is None:
            return ""
        try:
            res = mariadb_client.run_statement("SELECT DATABASE();")
            if mariadb_client.iserror():
                return ""
            if not res:
                return ""
            # try parsing HTML table via pandas
            try:
                dfs = pd.read_html(res)
                if dfs and len(dfs) > 0:
                    val = dfs[0].iloc[0, 0]
                    if isinstance(val, float) and pd.isna(val):
                        return ""
                    return str(val) if val is not None else ""
            except Exception:
                # fallback: regex extract first td
                m = re.search(r"<td.*?>(.*?)</td>", str(res), flags=re.S | re.I)
                if m:
                    txt = re.sub(r"<.*?>", "", m.group(1)).strip()
                    if txt.lower() == "null" or txt == "":
                        return ""
                    return txt
                txt = str(res).strip()
                if txt.lower() == "null" or txt == "":
                    return ""
                return txt
        except Exception:
            return ""
        return ""

    def _get_user_name(self, kernel):
        candidates = [
            getattr(kernel, "user_name", None),
            getattr(kernel, "username", None),
            getattr(kernel, "user", None),
            getattr(kernel, "session", None),
        ]
        for cand in candidates:
            if cand is None:
                continue
            if isinstance(cand, str) and cand.strip():
                return cand
            try:
                maybe = getattr(cand, "user", None)
                if isinstance(maybe, str) and maybe.strip():
                    return maybe
            except Exception:
                pass
        try:
            return os.getlogin()
        except Exception:
            return ""

    def _ensure_metadata_table(self, kernel, db_name):
        mariadb_client = self._get_mariadb_client(kernel)
        log = self._get_logger(kernel)
        if mariadb_client is None:
            return
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
                log.error("Error creating magic_metadata table.")
        except Exception as e:
            log.error(f"Failed to ensure magic_metadata table: {e}")

    def _insert_metadata(self, kernel, command_name, arguments, affected_columns,
                         operation_status, message, db_name, user_name):
        mariadb_client = self._get_mariadb_client(kernel)
        log = self._get_logger(kernel)
        if mariadb_client is None:
            return
        table_full_name = f"{db_name}.magic_metadata" if db_name else "magic_metadata"

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
            if mariadb_client.iserror():
                log.error("Error inserting into magic_metadata.")
        except Exception as e:
            log.error(f"Exception while inserting metadata: {e}")

    # --- end metadata helpers ---

    def execute(self, kernel, data):
        """Execute the dropoutliers magic (modifies data['last_select'] in-place) and log metadata."""
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

        # parse columns argument
        columns_arg = args.get("columns", None)
        if isinstance(columns_arg, str):
            columns = [c.strip() for c in columns_arg.split(",") if c.strip()]
        elif isinstance(columns_arg, (list, tuple)):
            columns = list(columns_arg)
        else:
            columns = None

        # method and params
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

        # Determine target numeric columns
        if columns is None:
            target_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        else:
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                kernel._send_message("stderr", f"Column(s) not found: {', '.join(missing_cols)}")
                # log metadata for failure and return
                try:
                    db_name = self._get_db_name(kernel)
                    user_name = self._get_user_name(kernel)
                    self._ensure_metadata_table(kernel, db_name)
                    self._insert_metadata(
                        kernel=kernel,
                        command_name=self.name(),
                        arguments=self.args if isinstance(self.args, str) else str(self.args),
                        affected_columns=",".join(columns) if columns else "",
                        operation_status="error",
                        message=f"Column(s) not found: {', '.join(missing_cols)}",
                        db_name=db_name,
                        user_name=user_name
                    )
                except Exception:
                    pass
                return
            # keep only numeric columns
            target_columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            non_numeric = [c for c in columns if c not in target_columns]
            if non_numeric:
                kernel._send_message("stdout", f"Warning: non-numeric columns skipped: {', '.join(non_numeric)}")

        if not target_columns:
            kernel._send_message("stderr", "No numeric target columns found to detect outliers.")
            # log metadata for early exit
            try:
                db_name = self._get_db_name(kernel)
                user_name = self._get_user_name(kernel)
                self._ensure_metadata_table(kernel, db_name)
                self._insert_metadata(
                    kernel=kernel,
                    command_name=self.name(),
                    arguments=self.args if isinstance(self.args, str) else str(self.args),
                    affected_columns="",
                    operation_status="error",
                    message="No numeric target columns found to detect outliers.",
                    db_name=db_name,
                    user_name=user_name
                )
            except Exception:
                pass
            return

        # Prepare metadata context
        db_name = self._get_db_name(kernel)
        user_name = self._get_user_name(kernel)
        # ensure metadata table exists
        try:
            self._ensure_metadata_table(kernel, db_name)
        except Exception:
            try:
                kernel._send_message("stdout", "Warning: failed to ensure metadata table (continuing).")
            except Exception:
                pass

        # Detect outliers per column and combine masks
        combined_mask = None
        messages = []
        operation_status = "success"
        try:
            for col in target_columns:
                try:
                    mask = self._detect_outliers_series(df[col], method, k=k, z_thresh=z_thresh)
                    n_out = int(mask.sum())
                    messages.append(f"Column '{col}': detected {n_out} outlier(s) using {method}.")
                    if combined_mask is None:
                        combined_mask = mask.astype(bool)
                    else:
                        combined_mask = combined_mask | mask.astype(bool)
                except Exception as e:
                    messages.append(f"Column '{col}': error detecting outliers: {e}")
        except Exception as e:
            operation_status = "error"
            messages.append(f"Fatal error while detecting outliers: {e}")

        # If no outliers found, log and return (but still record metadata)
        if combined_mask is None or not combined_mask.any():
            try:
                kernel._send_message("stdout", "No outliers detected. No rows removed.\n" + "\n".join(messages))
                self._send_html(kernel, df)
            except Exception:
                pass
            # insert metadata (no rows removed)
            try:
                self._insert_metadata(
                    kernel=kernel,
                    command_name=self.name(),
                    arguments=self.args if isinstance(self.args, str) else str(self.args),
                    affected_columns=", ".join(target_columns),
                    operation_status=operation_status,
                    message="\n".join(messages) or "No outliers detected.",
                    db_name=db_name,
                    user_name=user_name
                )
            except Exception:
                pass
            return

        # Drop rows in-place where any target column is an outlier
        try:
            n_before = len(df)
            df.drop(index=df[combined_mask].index, inplace=True)
            data["last_select"] = df
            n_after = len(df)
            removed = n_before - n_after
            kernel._send_message("stdout", f"Dropped {removed} row(s) containing outliers (in-place).\n" + "\n".join(messages))
            try:
                self._send_html(kernel, df)
            except Exception:
                pass
        except Exception as e:
            operation_status = "error"
            err = f"Error while removing outlier rows: {e}"
            kernel._send_message("stderr", err)
            messages.append(err)

        # Insert metadata (best-effort)
        try:
            self._insert_metadata(
                kernel=kernel,
                command_name=self.name(),
                arguments=self.args if isinstance(self.args, str) else str(self.args),
                affected_columns=", ".join(target_columns),
                operation_status=operation_status,
                message="\n".join(messages),
                db_name=db_name,
                user_name=user_name
            )
        except Exception:
            try:
                kernel._send_message("stdout", "Warning: failed to write metadata (continuing).")
            except Exception:
                pass
