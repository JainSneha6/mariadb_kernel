# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import logging
import os
import re

# Optional helper to reliably get current DB name (if available)
try:
    from mariadb_kernel.sql_fetch import SqlFetch
except Exception:
    SqlFetch = None


class FillMissing(MariaMagic):
    """
    %fillmissing [columns=col1,col2,...] [strategy=mean|median|mode|constant] [value=const]

    Always performs the operation IN-PLACE on data["last_select"]:

      - If columns provided, fill missing values only for those columns.
      - If no columns provided, fill missing values for all columns.
      - strategies:
          * mean    -> uses column mean (numeric columns only)
          * median  -> uses column median (numeric columns only)
          * mode    -> uses column mode (most frequent value; works for any dtype)
          * constant-> fills with provided value (value must be supplied via value=...)
    Execution metadata is recorded into table `magic_metadata`.
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "fillmissing"

    def help(self):
        return (
            "%fillmissing [columns=col1,col2,...] [strategy=mean|median|mode|constant] [value=const]\n"
            "Fills missing values in data['last_select'] (always IN-PLACE)."
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
            # Remove surrounding quotes if present so value="abc" becomes abc (still as string)
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

    # -------------------- metadata / DB helpers (best-effort) --------------------
    def _get_mariadb_client(self, kernel):
        """Return mariadb_client if present on kernel, else None"""
        return getattr(kernel, "mariadb_client", None)

    def _get_logger(self, kernel):
        """Return a logger on kernel if present, else create a temporary logger"""
        return getattr(kernel, "log", logging.getLogger(__name__))

    def _sql_escape(self, val):
        """Escape a value for SQL single-quoted literal insert. None -> NULL"""
        if val is None:
            return "NULL"
        if not isinstance(val, str):
            val = str(val)
        # double single-quotes for SQL escaping
        return "'" + val.replace("'", "''") + "'"

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
                log.debug("SqlFetch available but .get_db_name() failed; falling back.")

        # Fallback: run SELECT DATABASE();
        if mariadb_client is None:
            return ""
        try:
            result = mariadb_client.run_statement("SELECT DATABASE();")
            if mariadb_client.iserror():
                return ""
            if not result:
                return ""
            # If result is raw HTML table, try to parse with pandas
            try:
                df_list = pd.read_html(result)
                if df_list and isinstance(df_list, list) and len(df_list) > 0:
                    val = df_list[0].iloc[0, 0]
                    if isinstance(val, float) and pd.isna(val):
                        return ""
                    return str(val) if val is not None else ""
            except Exception:
                # if not parseable by pandas, try regex to extract first cell content
                m = re.search(r"<td.*?>(.*?)</td>", str(result), flags=re.S | re.I)
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

    def _get_user_name(self, kernel):
        """Try several places to find the current user name; fallback to OS login or empty string."""
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
                log.error("Error inserting into magic_metadata.")
        except Exception as e:
            log.error(f"Exception while inserting metadata: {e}")

    # -------------------- end metadata helpers --------------------

    def execute(self, kernel, data):
        """Execute the fillmissing magic (always modifies data['last_select']) and log metadata."""
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
            target_columns = [c.strip() for c in columns_arg.split(",") if c.strip()]
        elif isinstance(columns_arg, (list, tuple)):
            target_columns = list(columns_arg)
        else:
            target_columns = None

        # determine target columns (None => all columns)
        if target_columns is None:
            target_columns = list(df.columns)
        else:
            missing_cols = [c for c in target_columns if c not in df.columns]
            if missing_cols:
                kernel._send_message("stderr", f"Column(s) not found: {', '.join(missing_cols)}")
                # log metadata for failure
                try:
                    db_name = self._get_db_name(kernel)
                    user_name = self._get_user_name(kernel)
                    self._ensure_metadata_table(kernel, db_name)
                    self._insert_metadata(
                        kernel=kernel,
                        command_name=self.name(),
                        arguments=self.args if isinstance(self.args, str) else str(self.args),
                        affected_columns="\n".join(target_columns),
                        operation_status="error",
                        message=f"Column(s) not found: {', '.join(missing_cols)}",
                        db_name=db_name,
                        user_name=user_name
                    )
                except Exception:
                    pass
                return

        # parse strategy
        strategy = args.get("strategy", "mean")
        if isinstance(strategy, str):
            strategy = strategy.lower()
        else:
            strategy = str(strategy).lower()

        allowed = {"mean", "median", "mode", "constant"}
        if strategy not in allowed:
            kernel._send_message("stderr", f"Unknown strategy '{strategy}'. Allowed: {', '.join(allowed)}")
            return

        # constant requires value
        value_provided = "value" in args
        const_value = args.get("value", None)

        if strategy == "constant" and not value_provided:
            kernel._send_message("stderr", "Strategy 'constant' requires a 'value=...' argument.")
            return

        # Prepare metadata context and ensure table exists (best-effort)
        db_name = self._get_db_name(kernel)
        user_name = self._get_user_name(kernel)
        try:
            self._ensure_metadata_table(kernel, db_name)
        except Exception:
            try:
                kernel._send_message("stdout", "Warning: failed to ensure metadata table (continuing).")
            except Exception:
                pass

        # perform filling column by column with sensible handling for dtype
        messages = []
        operation_status = "success"
        for col in target_columns:
            try:
                series = df[col]
                if strategy in {"mean", "median"}:
                    # only numeric columns supported for mean/median
                    if pd.api.types.is_numeric_dtype(series):
                        if strategy == "mean":
                            fill_val = series.mean(skipna=True)
                        else:
                            fill_val = series.median(skipna=True)
                        # If result is NaN (e.g., all values missing), skip and warn
                        if pd.isna(fill_val):
                            messages.append(f"Column '{col}': no non-missing values to compute {strategy}. Skipped.")
                            continue
                        df[col].fillna(fill_val, inplace=True)
                        messages.append(f"Column '{col}': filled missing with {strategy}={fill_val}.")
                    else:
                        messages.append(f"Column '{col}' is not numeric; cannot use {strategy}. Skipped.")
                        continue

                elif strategy == "mode":
                    # mode works for any dtype; pick first mode if multiple
                    modes = series.mode(dropna=True)
                    if modes.empty:
                        messages.append(f"Column '{col}': no mode (all missing). Skipped.")
                        continue
                    fill_val = modes.iloc[0]
                    df[col].fillna(fill_val, inplace=True)
                    messages.append(f"Column '{col}': filled missing with mode={fill_val}.")

                elif strategy == "constant":
                    # use the parsed const_value directly
                    fill_val = const_value
                    df[col].fillna(fill_val, inplace=True)
                    messages.append(f"Column '{col}': filled missing with constant value={fill_val}.")

            except Exception as e:
                operation_status = "error"
                messages.append(f"Column '{col}': error while filling missing values: {e}")

        # update the data store and display results
        try:
            data["last_select"] = df
            summary = "\n".join(messages)
            kernel._send_message("stdout", f"Fill missing completed (in-place). Summary:\n{summary}")
            try:
                self._send_html(kernel, df)
            except Exception:
                pass
        except Exception as e:
            operation_status = "error"
            kernel._send_message("stderr", f"Error while updating last_select or displaying DataFrame: {e}")
            messages.append(f"Error while updating last_select or displaying DataFrame: {e}")

        # Insert metadata (best-effort)
        try:
            args_for_db = self.args if isinstance(self.args, str) else str(self.args)
            affected_columns_str = "\n".join(target_columns) if target_columns else ""
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
        except Exception:
            try:
                kernel._send_message("stdout", "Warning: failed to write metadata (continuing).")
            except Exception:
                pass
