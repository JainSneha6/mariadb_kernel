# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
from sklearn.model_selection import train_test_split


class SplitData(MariaMagic):
    """
    %splitdata [test_size=0.2] [val_size=0.1] [stratify=colname] [shuffle=True|False]
               [random_state=42] [inplace=True|False] [train_name=last_select_train]
               [test_name=last_select_test] [val_name=last_select_val]

    Split the current data["last_select"] DataFrame into train/test/(validation).

    - test_size: float fraction (0-1) or int count. Interpreted relative to the original dataset.
                 Default: 0.2
    - val_size:  float fraction (0-1) or int count. Interpreted relative to the original dataset.
                 If 0 (default), no validation set is created.
    - stratify:  column name to stratify on (must exist in the DataFrame).
    - shuffle:   whether to shuffle before splitting (default True).
    - random_state: integer seed for reproducibility (default None).
    - inplace:   if True (default), sets data["last_select"] to the training set and also stores
                 test/val under the provided names. If False, original last_select is kept and train/test/val
                 are stored under the provided names.
    - train_name/test_name/val_name: keys under which resulting DataFrames will be stored in `data`.
                 Defaults: last_select_train, last_select_test, last_select_val

    Behavior notes:
      - test_size and val_size may be integers (counts) or floats (fractions of the original dataset).
      - If both fractions are provided, the code first removes the test set (test_size of original),
        then splits the remaining data to create the validation set. The computed relative fraction
        for the second split uses val_size relative to the original dataset (so results match user intent).
      - If val_size is 0 or not provided, only train/test split occurs.

    Examples:
      %splitdata
      %splitdata test_size=0.25 val_size=0.1 stratify=target random_state=123
      %splitdata test_size=100 val_size=50 inplace=False
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "splitdata"

    def help(self):
        return (
            "%splitdata [test_size=0.2] [val_size=0.1] [stratify=colname] [shuffle=True|False]\n"
            "[random_state=42] [inplace=True|False] [train_name=name] [test_name=name] [val_name=name]\n"
            "Split last_select into train/test/(val)."
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

    def _send_html(self, kernel, df, title=None):
        try:
            html = df.to_html(index=False)
            if title:
                html = f"<h4>{title}</h4>" + html
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

        # Defaults
        test_size_arg = args.get("test_size", 0.2)
        val_size_arg = args.get("val_size", 0.0)
        stratify_col = args.get("stratify", None)
        shuffle = bool(args.get("shuffle", True))
        random_state = args.get("random_state", None)
        inplace = bool(args.get("inplace", True))

        train_name = args.get("train_name", "last_select_train")
        test_name = args.get("test_name", "last_select_test")
        val_name = args.get("val_name", "last_select_val")

        # Validate dataset
        n_total = len(df)
        if n_total == 0:
            kernel._send_message("stderr", "DataFrame has no rows to split.")
            return

        # Helper to interpret sizes (int count or fraction)
        def interpret_size(size_arg, total):
            if isinstance(size_arg, int):
                if size_arg < 0:
                    raise ValueError("Sizes must be non-negative.")
                return float(size_arg) / total
            try:
                size_f = float(size_arg)
            except Exception:
                raise ValueError("Size must be an int or float.")
            if size_f < 0:
                raise ValueError("Sizes must be non-negative.")
            if 0 <= size_f < 1:
                return size_f
            # If provided >=1 and integer-like, treat as count
            if size_f >= 1 and abs(size_f - int(size_f)) < 1e-9:
                if int(size_f) > total:
                    raise ValueError("Size count larger than dataset.")
                return float(int(size_f)) / total
            # fractions >= 1 are invalid
            raise ValueError("If numeric and >=1, size must be an integer count <= total rows.")

        try:
            test_frac = interpret_size(test_size_arg, n_total)
            val_frac = interpret_size(val_size_arg, n_total)
        except ValueError as e:
            kernel._send_message("stderr", f"Error interpreting sizes: {e}")
            return

        if test_frac + val_frac >= 1.0:
            kernel._send_message("stderr", "Sum of test_size and val_size must be less than 1.0.")
            return

        # Prepare stratify arrays if requested
        stratify_arr = None
        if stratify_col:
            if stratify_col not in df.columns:
                kernel._send_message("stderr", f"Stratify column '{stratify_col}' not found in DataFrame.")
                return
            stratify_arr = df[stratify_col].values

        try:
            # First split off the test set (test_frac of original)
            if test_frac > 0:
                train_val_df, test_df = train_test_split(
                    df,
                    test_size=test_frac,
                    shuffle=shuffle,
                    random_state=random_state,
                    stratify=stratify_arr if stratify_arr is not None else None
                )
            else:
                train_val_df = df.copy(deep=True)
                test_df = pd.DataFrame(columns=df.columns)

            # If no val requested, train = train_val_df
            if val_frac <= 0:
                train_df = train_val_df
                val_df = pd.DataFrame(columns=df.columns)
            else:
                # We need to compute val fraction relative to the remaining (train_val_df).
                # val_frac was relative to original; relative fraction = val_frac / (1 - test_frac)
                rel_val_frac = val_frac / (1.0 - test_frac)
                # For stratify on second split, use stratify column restricted to train_val_df if provided
                stratify_arr_second = None
                if stratify_arr is not None:
                    stratify_arr_second = train_val_df[stratify_col].values
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=rel_val_frac,
                    shuffle=shuffle,
                    random_state=random_state,
                    stratify=stratify_arr_second if stratify_arr_second is not None else None
                )

            # Store results in data dict
            data[test_name] = test_df
            data[val_name] = val_df
            data[train_name] = train_df

            if inplace:
                # follow behavior of other magics: set last_select to training set
                data["last_select"] = train_df

            # Report sizes
            msg = (
                f"Split completed: total={n_total}, train={len(train_df)}, "
                f"test={len(test_df)}, val={len(val_df)}."
            )
            kernel._send_message("stdout", msg)

            # Display small previews
            # Show train + validation (if exists) and test
            try:
                if not train_df.empty:
                    self._send_html(kernel, train_df.head(20), title=f"Train ({len(train_df)} rows)")
                if not val_df.empty:
                    self._send_html(kernel, val_df.head(20), title=f"Validation ({len(val_df)} rows)")
                if not test_df.empty:
                    self._send_html(kernel, test_df.head(20), title=f"Test ({len(test_df)} rows)")
            except Exception:
                # non-fatal; already stored the DataFrames
                pass

        except Exception as e:
            kernel._send_message("stderr", f"Error during splitting: {e}")
            return
