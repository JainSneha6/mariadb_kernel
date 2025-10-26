# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd


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
    Examples:
      %fillmissing
        -> fills numeric columns with their mean (default strategy=mean)
      %fillmissing columns=age,salary strategy=median
        -> fills age and salary missing values with column medians (in-place)
      %fillmissing columns=name strategy=constant value="unknown"
        -> fills name with "unknown" where missing (in-place)
      %fillmissing strategy=mode
        -> fills every column's missing values with its mode (if exists)
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

    def execute(self, kernel, data):
        """Execute the fillmissing magic (always modifies data['last_select'])."""
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

        # determine target columns (None => all columns)
        if columns is None:
            target_columns = list(df.columns)
        else:
            target_columns = columns
            missing_cols = [c for c in target_columns if c not in df.columns]
            if missing_cols:
                kernel._send_message("stderr", f"Column(s) not found: {', '.join(missing_cols)}")
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

        # perform filling column by column with sensible handling for dtype
        messages = []
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
                    # If fill_val is a string that looks like "None", we want to keep it as string;
                    # do not coerce types implicitly — user controls value type via quotes or unquoted numbers.
                    df[col].fillna(fill_val, inplace=True)
                    messages.append(f"Column '{col}': filled missing with constant value={fill_val}.")

            except Exception as e:
                messages.append(f"Column '{col}': error while filling missing values: {e}")

        # update the data store and display results
        try:
            data["last_select"] = df
            summary = "\n".join(messages)
            kernel._send_message("stdout", f"Fill missing completed (in-place). Summary:\n{summary}")
            self._send_html(kernel, df)
        except Exception as e:
            kernel._send_message("stderr", f"Error while updating last_select or displaying DataFrame: {e}")
