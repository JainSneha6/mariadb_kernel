# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import pandas as pd
import shlex
from distutils import util


class Stats(MariaMagic):
    """
    %stats [columns=col1,col2] [include=all|numeric|object] [percentiles=25,50,75] [transpose=true|false]

    Produce a statistical summary of the DataFrame in data["last_select"].

    Examples:
      %stats
        -> numeric summary (count, mean, std, min, 25%, 50%, 75%, max)
      %stats include=all
        -> include all dtypes (object, category, datetime etc.)
      %stats columns=age,salary
        -> summary only for the specified columns
      %stats percentiles=10,90
        -> include the 10th and 90th percentiles (values can be 0-100 or 0-1)
      %stats transpose=true
        -> show summary transposed (rows <-> columns)
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "stats"

    def help(self):
        return (
            "%stats [columns=col1,col2] [include=all|numeric|object] "
            "[percentiles=25,50,75] [transpose=true|false]\n"
            "Show statistical summary (uses pandas.DataFrame.describe under the hood)."
        )

    def _str_to_obj(self, s):
        """Cast string tokens to int/float/bool if possible, otherwise return string."""
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
        """Parse arguments given as key=value pairs (space separated)."""
        if not input_str or input_str.strip() == "":
            return {}
        pairs = dict(token.split("=", 1) for token in shlex.split(input_str))
        for k, v in pairs.items():
            pairs[k] = self._str_to_obj(v)
        return pairs

    def _send_html(self, kernel, df):
        """Send a DataFrame as HTML (fallback to plain text)."""
        try:
            html = df.to_html()
            mime = "text/html"
        except Exception:
            html = str(df)
            mime = "text/plain"
        display_content = {"data": {mime: html}, "metadata": {}}
        kernel.send_response(kernel.iopub_socket, "display_data", display_content)

    def _parse_percentiles(self, pct_arg):
        """
        Accept percentiles as comma-separated list of numbers.
        Values may be 0-100 (e.g. 25) or 0-1 (e.g. 0.25).
        Return list of floats in [0,1] as required by pandas.
        """
        if pct_arg is None:
            return None
        if isinstance(pct_arg, (list, tuple)):
            raw = pct_arg
        else:
            raw = str(pct_arg).split(",")
        out = []
        for item in raw:
            s = str(item).strip()
            if s == "":
                continue
            try:
                v = float(s)
            except ValueError:
                # ignore bad token
                continue
            if v > 1:
                v = v / 100.0
            if 0 <= v <= 1:
                out.append(v)
        # pandas.describe requires percentiles to be sorted and unique
        out = sorted(set(out))
        return out if out else None

    def execute(self, kernel, data):
        """Execute the %stats magic (display-only)."""
        df = data.get("last_select")
        if df is None:
            kernel._send_message("stderr", "No last_select found in kernel data.")
            return

        if hasattr(df, "empty") and df.empty:
            kernel._send_message("stderr", "There is no data to summarize (empty DataFrame).")
            return

        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return

        # columns handling
        cols_arg = args.get("columns", None)
        if isinstance(cols_arg, str):
            columns = [c.strip() for c in cols_arg.split(",") if c.strip()]
        elif isinstance(cols_arg, (list, tuple)):
            columns = list(cols_arg)
        else:
            columns = None

        # include: pandas.describe 'include' parameter (None default -> numeric)
        include = args.get("include", "numeric")
        if include not in ("numeric", "object", "all"):
            # allow user to pass pandas dtypes-like include, but restrict to these for simplicity
            include = "numeric"
        include_param = None
        if include == "all":
            include_param = "all"
        elif include == "object":
            include_param = object
        else:
            include_param = None  # pandas default -> numeric only

        # percentiles
        percentiles_arg = args.get("percentiles", None)
        percentiles = self._parse_percentiles(percentiles_arg)

        transpose = bool(args.get("transpose", False))

        # subset dataframe if columns specified
        try:
            subdf = df[columns] if columns is not None else df
        except KeyError as e:
            kernel._send_message("stderr", f"Column not found: {e}")
            return

        # call pandas describe
        try:
            describe_kwargs = {}
            if percentiles is not None:
                describe_kwargs["percentiles"] = percentiles
            if include_param is not None:
                describe_kwargs["include"] = include_param

            result = subdf.describe(**describe_kwargs)
            # For object dtypes, pandas describe may include top/freq; that's fine.
            if transpose:
                try:
                    result = result.transpose()
                except Exception:
                    # fallback without transposing if something goes wrong
                    pass

            self._send_html(kernel, result)
        except Exception as e:
            kernel._send_message("stderr", f"Error computing statistics: {e}")
