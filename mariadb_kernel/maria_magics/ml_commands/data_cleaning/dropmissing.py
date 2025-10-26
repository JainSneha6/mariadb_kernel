# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd


class DropMissing(MariaMagic):
    """
    %dropmissing [columns=col1,col2,...]

    Always performs the operation IN-PLACE on data["last_select"]:
      - If columns are provided, drop rows where any of those columns is missing.
      - If no columns provided, drop rows that have any missing value (any column).

    Examples:
      %dropmissing
        -> drop rows with any missing value (in-place)
      %dropmissing columns=age
        -> drop rows where 'age' is missing (in-place)
      %dropmissing columns=age,salary
        -> drop rows where 'age' OR 'salary' is missing (in-place)
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "dropmissing"

    def help(self):
        return (
            "%dropmissing [columns=col1,col2,...]\n"
            "Drops rows with missing values from data['last_select'] (always IN-PLACE)."
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

    def execute(self, kernel, data):
        """Execute the dropmissing magic (always modifies data['last_select'])."""
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

        columns_arg = args.get("columns", None)
        if isinstance(columns_arg, str):
            columns = [c.strip() for c in columns_arg.split(",") if c.strip()]
        elif isinstance(columns_arg, (list, tuple)):
            columns = list(columns_arg)
        else:
            columns = None

        if columns is not None:
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                kernel._send_message("stderr", f"Column(s) not found: {', '.join(missing_cols)}")
                return

        try:
            df.dropna(axis=0, subset=columns, inplace=True)
            data["last_select"] = df
            kernel._send_message("stdout", "Dropped rows with missing values (in-place). Updated last_select.")
            self._send_html(kernel, df)
        except Exception as e:
            kernel._send_message("stderr", f"Error while dropping missing values: {e}")
