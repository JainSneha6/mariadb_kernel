# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np


class DropOutliers(MariaMagic):
    """
    %dropoutliers [columns=col1,col2,...] [method=iqr|zscore] [k=1.5] [z_thresh=3.0]

    Removes rows (IN-PLACE) from data['last_select'] where any selected numeric column
    is detected as an outlier according to the chosen method.

    - method:
        iqr    -> Tukey IQR method using k (default 1.5)
        zscore -> absolute z-score above z_thresh (default 3.0)

    Examples:
      %dropoutliers
        -> use IQR with k=1.5 on all numeric columns and drop rows containing outliers
      %dropoutliers columns=age,salary method=zscore z_thresh=2.5
        -> drop rows where age OR salary has |z| > 2.5
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
            "Removes rows containing outliers from data['last_select'] (in-place)."
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

    def execute(self, kernel, data):
        """Execute the dropoutliers magic (modifies data['last_select'] in-place)."""
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
                return
            # keep only numeric columns
            target_columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            non_numeric = [c for c in columns if c not in target_columns]
            if non_numeric:
                kernel._send_message("stdout", f"Warning: non-numeric columns skipped: {', '.join(non_numeric)}")

        if not target_columns:
            kernel._send_message("stderr", "No numeric target columns found to detect outliers.")
            return

        # Detect outliers per column and combine masks
        combined_mask = None
        messages = []
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

        if combined_mask is None or not combined_mask.any():
            kernel._send_message("stdout", "No outliers detected. No rows removed.\n" + "\n".join(messages))
            # still show DataFrame
            try:
                self._send_html(kernel, df)
            except Exception:
                pass
            return

        # Drop rows in-place where any target column is an outlier
        try:
            n_before = len(df)
            df.drop(index=df[combined_mask].index, inplace=True)
            data["last_select"] = df
            n_after = len(df)
            kernel._send_message("stdout", f"Dropped {n_before - n_after} row(s) containing outliers (in-place).\n" + "\n".join(messages))
            try:
                self._send_html(kernel, df)
            except Exception:
                pass
        except Exception as e:
            kernel._send_message("stderr", f"Error while removing outlier rows: {e}")
