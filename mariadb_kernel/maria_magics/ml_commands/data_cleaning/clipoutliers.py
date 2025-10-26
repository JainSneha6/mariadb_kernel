# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np


class ClipOutliers(MariaMagic):
    """
    %clipoutliers [columns=col1,col2,...] [method=iqr|zscore]
                 [k=1.5] [z_thresh=3.0] [inplace=True|False]

    Clamps (clips) extreme values to computed boundary limits.

    - method:
        iqr        -> Tukey IQR method using k (default 1.5)
        zscore     -> mean ± z_thresh * std (default z_thresh=3.0)

    - columns: comma-separated list of columns to operate on. If omitted, all numeric columns are used.
    - inplace: if True (default) modifies data["last_select"] in-place.
               if False stores clipped copy in data["last_select_clipped"].

    Examples:
      %clipoutliers                 -> clip numeric columns using iqr (k=1.5) in-place
      %clipoutliers method=zscore z_thresh=2.5 columns=age,salary inplace=False
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

    def execute(self, kernel, data):
        """Execute the %clipoutliers magic."""
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

        target_df = df if inplace else df.copy(deep=True)

        messages = []
        total_clipped = 0

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
                target_df[col] = series.clip(lower=lower, upper=upper)
                total_clipped += n_changed
                messages.append(f"Column '{col}': clipped {n_changed} value(s) (bounds: {lower:.4f}, {upper:.4f}).")
            except Exception as e:
                messages.append(f"Column '{col}': error while clipping: {e}")

        if inplace:
            data["last_select"] = target_df
            location_msg = "Modified in-place: data['last_select'] updated."
        else:
            data["last_select_clipped"] = target_df
            location_msg = "Result stored in data['last_select_clipped'] (original unchanged)."

        kernel._send_message("stdout", f"Clip outliers completed using {method}.\n"
                                       + "\n".join(messages)
                                       + f"\nTotal values clipped: {total_clipped}. {location_msg}")

        # Show output
        try:
            self._send_html(kernel, target_df)
        except Exception:
            pass
