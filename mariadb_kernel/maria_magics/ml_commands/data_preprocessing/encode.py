# encode.py
# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import pandas as pd
import shlex
from distutils import util
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import logging
import os
import re

# Optional helper to reliably get current DB name (if available)
try:
    from mariadb_kernel.sql_fetch import SqlFetch
except Exception:
    SqlFetch = None


class Encode(MariaMagic):
    """
    %encode method=<label|onehot|ordinal>
            [columns=col1,col2,...]
            [inplace=true|false]
            [drop_original=true|false]

    Notes:
     - If columns omitted, object/category dtype columns are auto-selected.
     - Default: inplace=true, drop_original=true.
     - Requires scikit-learn installed for onehot/ordinal.
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "encode"

    def help(self):
        return (
            "%encode method=<label|onehot|ordinal> [columns=col1,col2] "
            "[inplace=true] [drop_original=true]\n"
            "Encode categorical columns using label, one-hot, or ordinal encoding (automatic)."
            "Execution metadata is recorded in table `magic_metadata`."
        )

    def _str_to_obj(self, s):
        """Cast to int/float/bool when possible, otherwise return string."""
        try:
            return int(s)
        except (ValueError, TypeError):
            pass
        try:
            return float(s)
        except (ValueError, TypeError):
            pass
        try:
            return bool(util.strtobool(str(s)))
        except Exception:
            return s

    def parse_args(self, input_str):
        if not input_str or input_str.strip() == "":
            return {}
        pairs = dict(token.split("=", 1) for token in shlex.split(input_str))
        for k, v in pairs.items():
            pairs[k] = self._str_to_obj(v)
        return pairs

    def _send_html(self, kernel, df):
        try:
            html = df.to_html(index=False)
            mime = "text/html"
        except Exception:
            html = str(df)
            mime = "text/plain"
        display_content = {"data": {mime: html}, "metadata": {}}
        kernel.send_response(kernel.iopub_socket, "display_data", display_content)

    def _make_ohe(self, **kwargs):
        """
        Create OneHotEncoder in a sklearn-version compatible way.
        Older sklearn versions accept `sparse`; newer use `sparse_output`.
        """
        try:
            return OneHotEncoder(sparse=False, **kwargs)
        except TypeError:
            # fallback for newer sklearn where parameter name changed
            return OneHotEncoder(sparse_output=False, **kwargs)

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
        return "'" + val.replace("'", "''") + "'"

    def _get_db_name(self, kernel):
        """
        Attempt to determine the currently used DB.
        Prefer SqlFetch if available; otherwise run SELECT DATABASE(); and try to parse.
        Returns empty string if none found.
        """
        mariadb_client = self._get_mariadb_client(kernel)
        log = self._get_logger(kernel)

        if SqlFetch is not None and mariadb_client is not None:
            try:
                sf = SqlFetch(mariadb_client, log)
                dbname = sf.get_db_name()
                if isinstance(dbname, str):
                    return dbname
            except Exception:
                log.debug("SqlFetch available but .get_db_name() failed; falling back.")

        if mariadb_client is None:
            return ""
        try:
            result = mariadb_client.run_statement("SELECT DATABASE();")
            if mariadb_client.iserror():
                return ""
            if not result:
                return ""
            # Try to parse HTML table with pandas
            try:
                dfs = pd.read_html(result)
                if dfs and len(dfs) > 0:
                    val = dfs[0].iloc[0, 0]
                    if isinstance(val, float) and pd.isna(val):
                        return ""
                    return str(val) if val is not None else ""
            except Exception:
                m = re.search(r"<td.*?>(.*?)</td>", str(result), flags=re.S | re.I)
                if m:
                    txt = re.sub(r"<.*?>", "", m.group(1)).strip()
                    if txt.lower() == "null" or txt == "":
                        return ""
                    return txt
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

    # -------------------- end metadata helpers --------------------

    def execute(self, kernel, data):
        # get DataFrame
        df = data.get("last_select")
        if df is None:
            kernel._send_message("stderr", "No last_select found in kernel data.")
            return
        if hasattr(df, "empty") and df.empty:
            kernel._send_message("stderr", "There is no data to encode (empty DataFrame).")
            return

        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments.")
            return

        method = str(args.get("method", "label")).lower()
        cols_arg = args.get("columns", None)
        if isinstance(cols_arg, str):
            columns = [c.strip() for c in cols_arg.split(",") if c.strip()]
        elif isinstance(cols_arg, (list, tuple)):
            columns = list(cols_arg)
        else:
            columns = list(df.select_dtypes(include=["object", "category"]).columns)

        if not columns:
            kernel._send_message("stderr", "No columns specified or detected for encoding.")
            # log metadata for failure
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
                    message="No columns specified or detected for encoding.",
                    db_name=db_name,
                    user_name=user_name
                )
            except Exception:
                pass
            return

        # validate existence
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            msg = f"Column(s) not found: {', '.join(missing_cols)}"
            kernel._send_message("stderr", msg)
            # log metadata for failure
            try:
                db_name = self._get_db_name(kernel)
                user_name = self._get_user_name(kernel)
                self._ensure_metadata_table(kernel, db_name)
                self._insert_metadata(
                    kernel=kernel,
                    command_name=self.name(),
                    arguments=self.args if isinstance(self.args, str) else str(self.args),
                    affected_columns="\n".join(columns),
                    operation_status="error",
                    message=msg,
                    db_name=db_name,
                    user_name=user_name
                )
            except Exception:
                pass
            return

        inplace = bool(args.get("inplace", True))
        drop_original = bool(args.get("drop_original", True))

        # Work on copy if not inplace
        result_df = df if inplace else df.copy()

        # Prepare metadata context
        db_name = self._get_db_name(kernel)
        user_name = self._get_user_name(kernel)
        try:
            self._ensure_metadata_table(kernel, db_name)
        except Exception:
            try:
                kernel._send_message("stdout", "Warning: failed to ensure metadata table (continuing).")
            except Exception:
                pass

        messages = []
        operation_status = "success"
        created_columns = []

        try:
            # We'll store encoder info here to save into data at the end
            encoder_obj = None
            label_mappings = None

            if method == "label":
                # Use pandas.factorize which handles NaN by assigning -1 codes
                label_mappings = {}
                for col in columns:
                    codes, uniques = pd.factorize(result_df[col], sort=True)
                    new_col = f"{col}_lbl"
                    result_df[new_col] = codes
                    created_columns.append(new_col)
                    # Save mapping value->code for reuse later
                    mapping = {val: idx for idx, val in enumerate(uniques)}
                    label_mappings[col] = mapping
                    if drop_original:
                        result_df.drop(columns=[col], inplace=True)
                    messages.append(f"Column '{col}': label-encoded -> {new_col} (unique_values={len(uniques)})")

                encoder_obj = label_mappings

            elif method == "onehot":
                # sklearn OneHotEncoder with version compatibility
                encoder = self._make_ohe(handle_unknown="ignore")
                # replace NaN with sentinel string so it's treated as a category
                tmp = result_df[columns].astype(object).fillna("___MISSING___")
                arr = encoder.fit_transform(tmp)
                # feature names (sklearn >= 1.0)
                try:
                    feature_names = encoder.get_feature_names_out(columns)
                    feature_names = [str(fn) for fn in feature_names]
                except Exception:
                    # fallback: build names manually
                    cats = encoder.categories_
                    feature_names = []
                    for cname, cat_list in zip(columns, cats):
                        for cat in cat_list:
                            feature_names.append(f"{cname}_{str(cat)}")
                # create DataFrame of encoded features
                ohe_df = pd.DataFrame(arr, columns=feature_names, index=result_df.index)
                # concatenate appropriately
                if drop_original:
                    result_df = pd.concat([result_df.drop(columns=columns), ohe_df], axis=1)
                else:
                    result_df = pd.concat([result_df, ohe_df], axis=1)
                created_columns.extend(feature_names)
                messages.append(f"Columns {columns} one-hot encoded -> created {len(feature_names)} columns.")
                encoder_obj = encoder  # save fitted OneHotEncoder

            elif method == "ordinal":
                # use sklearn OrdinalEncoder for one or multiple columns (automatic ordering)
                enc = OrdinalEncoder(dtype=np.float64)
                # fillna sentinel so OrdinalEncoder treats missing as a category
                tmp = result_df[columns].astype(object).fillna("___MISSING___")
                enc_arr = enc.fit_transform(tmp)
                for i, col in enumerate(columns):
                    new_col = f"{col}_ord"
                    result_df[new_col] = enc_arr[:, i]
                    created_columns.append(new_col)
                    if drop_original:
                        result_df.drop(columns=[col], inplace=True)
                    messages.append(f"Column '{col}': ordinal-encoded -> {new_col}")
                encoder_obj = enc

            else:
                kernel._send_message("stderr", "Unsupported method. Supported: label, onehot, ordinal.")
                # log unsupported method
                try:
                    self._insert_metadata(
                        kernel=kernel,
                        command_name=self.name(),
                        arguments=self.args if isinstance(self.args, str) else str(self.args),
                        affected_columns="\n".join(columns),
                        operation_status="error",
                        message="Unsupported method requested.",
                        db_name=db_name,
                        user_name=user_name
                    )
                except Exception:
                    pass
                return

            # Apply result back to shared data if inplace
            if inplace:
                data["last_select"] = result_df
                kernel._send_message("stdout", "Encoded columns in-place and updated last_select.")
            else:
                kernel._send_message("stdout", "Displayed encoded result (last_select not modified).")

            # Save encoder (or mapping) to shared data for downstream pipeline usage
            try:
                if encoder_obj is not None:
                    data["last_select_encoder"] = encoder_obj
                elif label_mappings is not None:
                    data["last_select_encoder"] = label_mappings
            except Exception:
                # don't fail pipeline just because we couldn't save encoder
                pass

            # display
            self._send_html(kernel, result_df)

        except Exception as e:
            operation_status = "error"
            err_msg = f"Error during encoding: {e}"
            kernel._send_message("stderr", err_msg)
            messages.append(err_msg)

        # Attempt to insert metadata (best-effort)
        try:
            args_for_db = self.args if isinstance(self.args, str) else str(self.args)
            # store affected (input) columns newline-separated
            affected_columns_str = "\n".join(columns)
            # store created columns newline-separated (if any)
            created_columns_str = "\n".join(created_columns) if created_columns else ""
            # Compose metadata message with sections for readability
            details = "\n".join(messages) if messages else "Encoding completed without detailed messages."
            metadata_message = f"Method: {method}\nCreated columns:\n{created_columns_str}\n\nDetails:\n{details}"
            self._insert_metadata(
                kernel=kernel,
                command_name=self.name(),
                arguments=args_for_db,
                affected_columns=affected_columns_str,
                operation_status=operation_status,
                message=metadata_message,
                db_name=db_name,
                user_name=user_name
            )
        except Exception as e:
            try:
                kernel._send_message("stdout", f"Warning: failed to write metadata: {e}")
            except Exception:
                pass
