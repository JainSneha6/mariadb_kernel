# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import pandas as pd
import shlex
from distutils import util


class Missing(MariaMagic):
    """
    %missing [action=show|percent|summary] [columns=col1,col2]
    
    Examples:
      %missing                         -> shows count+percent of missing for all columns
      %missing action=percent          -> shows percent only
      %missing action=summary          -> shows dtype, missing, percent
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "missing"

    def help(self):
        return (
            "%missing [action=show|percent|summary] [columns=col1,col2]\n"
            "Display missing-value information from the last query result."
        )

    def _str_to_obj(self, s):
        """Cast strings to Python objects where possible."""
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                pass
        try:
            return bool(util.strtobool(s))
        except ValueError:
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
        """Display DataFrame as HTML in the notebook."""
        try:
            html = df.to_html()
            mime = "text/html"
        except Exception:
            html = str(df)
            mime = "text/plain"

        display_content = {"data": {mime: html}, "metadata": {}}
        kernel.send_response(kernel.iopub_socket, "display_data", display_content)

    def execute(self, kernel, data):
        """Main execution for %missing magic."""
        df = data.get("last_select")
        if df is None or (hasattr(df, "empty") and df.empty):
            kernel._send_message("stderr", "No data available to inspect for missing values.")
            return

        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments.")
            return

        action = args.get("action", "show")
        cols_arg = args.get("columns", None)

        if isinstance(cols_arg, str):
            columns = [c.strip() for c in cols_arg.split(",") if c.strip()]
        elif isinstance(cols_arg, (list, tuple)):
            columns = list(cols_arg)
        else:
            columns = None

        try:
            subdf = df[columns] if columns else df
        except KeyError as e:
            kernel._send_message("stderr", f"Column not found: {e}")
            return

        # Compute missing information
        missing_counts = subdf.isnull().sum()
        total = len(subdf)
        percent = (missing_counts / total * 100).round(2)

        out = pd.DataFrame({"missing": missing_counts, "percent": percent})
        if action == "percent":
            out = out[["percent"]]
        elif action == "summary":
            out["dtype"] = subdf.dtypes.astype(str)
            out = out[["dtype", "missing", "percent"]]

        self._send_html(kernel, out)
