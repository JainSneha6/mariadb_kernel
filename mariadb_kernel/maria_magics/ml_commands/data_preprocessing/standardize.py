# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Standardize(MariaMagic):
    """
    %standardize [columns=col1,col2,...] [inplace=True|False]

    Standardizes numeric columns using sklearn's StandardScaler
    (zero mean and unit variance).

    - columns: comma-separated list of columns to standardize.
               If omitted, all numeric columns are used.
    - inplace: if True (default), modifies data["last_select"] in-place.
               if False, stores result in data["last_select_standardized"].

    Examples:
      %standardize
      %standardize columns=age,salary inplace=False
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "standardize"

    def help(self):
        return (
            "%standardize [columns=col1,col2,...] [inplace=True|False]\n"
            "Standardizes numeric columns using sklearn's StandardScaler (in-place by default)."
        )

    def _str_to_obj(self, s):
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
            if isinstance(s, str) and len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                return s[1:-1]
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
            kernel.send_response(kernel.iopub_socket, "display_data",
                                 {"data": {"text/html": html}, "metadata": {}})
        except Exception:
            pass

    def execute(self, kernel, data):
        df = data.get("last_select")
        if df is None or df.empty:
            kernel._send_message("stderr", "No last_select found or DataFrame is empty.")
            return

        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return

        columns_arg = args.get("columns", None)
        if isinstance(columns_arg, str):
            columns = [c.strip() for c in columns_arg.split(",") if c.strip()]
        else:
            columns = None

        inplace = bool(args.get("inplace", True))
        target_df = df if inplace else df.copy(deep=True)

        if columns is None:
            target_columns = [c for c in target_df.columns if pd.api.types.is_numeric_dtype(target_df[c])]
        else:
            missing_cols = [c for c in columns if c not in target_df.columns]
            if missing_cols:
                kernel._send_message("stderr", f"Missing columns: {', '.join(missing_cols)}")
                return
            target_columns = columns

        if not target_columns:
            kernel._send_message("stderr", "No numeric columns to standardize.")
            return

        try:
            scaler = StandardScaler()
            target_df[target_columns] = scaler.fit_transform(target_df[target_columns])
            msg = f"Standardized {len(target_columns)} column(s) (mean=0, std=1)."
        except Exception as e:
            kernel._send_message("stderr", f"Error during standardization: {e}")
            return

        if inplace:
            data["last_select"] = target_df
            msg += " Updated data['last_select'] in-place."
        else:
            data["last_select_standardized"] = target_df
            msg += " Stored in data['last_select_standardized']."

        kernel._send_message("stdout", msg)
        self._send_html(kernel, target_df)
