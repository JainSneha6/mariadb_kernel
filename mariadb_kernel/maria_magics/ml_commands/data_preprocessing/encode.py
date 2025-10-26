# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import pandas as pd
import shlex
from distutils import util
import numpy as np

# sklearn imports (we'll create encoder instances in a version-compatible way)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


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
            return

        # validate existence
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            kernel._send_message("stderr", f"Column(s) not found: {', '.join(missing_cols)}")
            return

        inplace = bool(args.get("inplace", True))
        drop_original = bool(args.get("drop_original", True))

        # Work on copy if not inplace
        result_df = df if inplace else df.copy()

        try:
            if method == "label":
                # Use pandas.factorize which handles NaN by assigning -1 codes
                for col in columns:
                    codes, uniques = pd.factorize(result_df[col], sort=True)
                    new_col = f"{col}_lbl"
                    result_df[new_col] = codes
                    if drop_original:
                        result_df.drop(columns=[col], inplace=True)

            elif method == "onehot":
                # sklearn OneHotEncoder with version compatibility
                encoder = self._make_ohe(handle_unknown="ignore")
                # replace NaN with sentinel string so it's treated as a category
                arr = encoder.fit_transform(result_df[columns].astype(object).fillna("___MISSING___"))
                # feature names (sklearn >= 1.0)
                try:
                    feature_names = encoder.get_feature_names_out(columns)
                except Exception:
                    # fallback: build names manually
                    cats = encoder.categories_
                    feature_names = []
                    for cname, cat_list in zip(columns, cats):
                        for cat in cat_list:
                            feature_names.append(f"{cname}_{str(cat)}")
                ohe_df = pd.DataFrame(arr, columns=feature_names, index=result_df.index)
                if drop_original:
                    result_df = pd.concat([result_df.drop(columns=columns), ohe_df], axis=1)
                else:
                    result_df = pd.concat([result_df, ohe_df], axis=1)

            elif method == "ordinal":
                # use sklearn OrdinalEncoder for one or multiple columns (automatic ordering)
                enc = OrdinalEncoder(dtype=np.float64)
                # fillna sentinel so OrdinalEncoder treats missing as a category
                tmp = result_df[columns].astype(object).fillna("___MISSING___")
                enc_arr = enc.fit_transform(tmp)
                for i, col in enumerate(columns):
                    result_df[f"{col}_ord"] = enc_arr[:, i]
                    if drop_original:
                        result_df.drop(columns=[col], inplace=True)

            else:
                kernel._send_message("stderr", "Unsupported method. Supported: label, onehot, ordinal.")
                return

            # Apply result
            if inplace:
                data["last_select"] = result_df
                kernel._send_message("stdout", "Encoded columns in-place and updated last_select.")
            else:
                kernel._send_message("stdout", "Displayed encoded result (last_select not modified).")

            # display
            self._send_html(kernel, result_df)

        except Exception as e:
            kernel._send_message("stderr", f"Error during encoding: {e}")
            return
