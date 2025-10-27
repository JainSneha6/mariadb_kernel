# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import pandas as pd
import numpy as np
import shlex
import json
from distutils import util


class Predict(MariaMagic):
    """
    %predict_model model_name=last_model data_name=last_select_test output_name=last_preds
                   [show_cols=10] [proba=True|False]

    You can also provide inline values:
      %predict_model model_name=last_model data_name=[38, 80000.0] output_name=last_preds
    """

    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "predict_model"

    def help(self):
        return "Run predictions using a trained model stored in data[model_name], with optional inline feature values."

    def _str_to_obj(self, s):
        # try to interpret numbers, booleans, lists, or JSON
        try:
            return int(s)
        except Exception:
            pass
        try:
            return float(s)
        except Exception:
            pass
        try:
            return bool(util.strtobool(s))
        except Exception:
            pass
        try:
            return json.loads(s)
        except Exception:
            pass
        # strip quotes
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

    def execute(self, kernel, data):
        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return

        model_name = args.get("model_name", "last_model")
        data_arg = args.get("data_name", "last_select_test")
        output_name = args.get("output_name", "last_preds")
        show_cols = int(args.get("show_cols", 10))
        show_proba = bool(args.get("proba", False))

        # --- 1. Retrieve model ---
        model = data.get(model_name)
        if model is None:
            kernel._send_message("stderr", f"No model found in data['{model_name}']. Train one first.")
            return

        # --- 2. Load metadata ---
        meta = data.get(model_name + "_meta", {})
        features = meta.get("features")
        problem = meta.get("problem", "regression")

        if not features:
            kernel._send_message("stderr", "Model meta missing 'features'. Using numeric columns only if applicable.")
            features = []

        # --- 3. Determine input mode ---
        df = None
        if isinstance(data_arg, list):
            # Inline list of feature values
            if not features:
                kernel._send_message("stderr", "Cannot use inline values: model has no stored feature names.")
                return
            if len(data_arg) != len(features):
                kernel._send_message("stderr", f"Number of values ({len(data_arg)}) doesn't match expected features ({len(features)}): {features}")
                return
            df = pd.DataFrame([data_arg], columns=features)
            kernel._send_message("stdout", f"Using inline feature values for prediction: {dict(zip(features, data_arg))}")
        elif isinstance(data_arg, str) and data_arg.startswith("[") and data_arg.endswith("]"):
            # If user passed JSON array as string, parse it
            try:
                vals = json.loads(data_arg)
                if len(vals) != len(features):
                    kernel._send_message("stderr", f"Number of values ({len(vals)}) doesn't match expected features ({len(features)}): {features}")
                    return
                df = pd.DataFrame([vals], columns=features)
                kernel._send_message("stdout", f"Using inline feature values for prediction: {dict(zip(features, vals))}")
            except Exception as e:
                kernel._send_message("stderr", f"Error parsing inline data list: {e}")
                return
        else:
            # DataFrame-based mode
            df = data.get(data_arg)
            if df is None or df.empty:
                kernel._send_message("stderr", f"No DataFrame found in data['{data_arg}'] or it's empty.")
                return

        # --- 4. Align columns to features ---
        df_cols = df.columns.tolist()
        missing = [c for c in features if c not in df_cols]
        extra = [c for c in df_cols if c not in features]

        if missing:
            kernel._send_message("stderr", f"Missing columns not in input: {missing}. Filling with zeros.")
        if extra:
            kernel._send_message("stderr", f"Ignoring extra columns not seen during training: {extra}.")

        X = pd.DataFrame({col: df[col] if col in df.columns else 0 for col in features})

        # --- 5. Run predictions ---
        try:
            if show_proba and problem == "classification" and hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)
                if hasattr(model, "classes_"):
                    class_labels = [str(c) for c in model.classes_]
                    pred_df = pd.DataFrame(preds, columns=[f"proba_{c}" for c in class_labels])
                else:
                    pred_df = pd.DataFrame(preds, columns=[f"proba_{i}" for i in range(preds.shape[1])])
            else:
                y_pred = model.predict(X)
                pred_df = pd.DataFrame(y_pred, columns=["prediction"])
        except Exception as e:
            kernel._send_message("stderr", f"Error during prediction: {e}")
            return

        # --- 6. Save & display ---
        data[output_name] = pred_df

        try:
            kernel._send_html(pred_df.head(show_cols), title=f"Predictions ({output_name})")
        except Exception:
            kernel._send_message("stdout", pred_df.head(show_cols).to_string(index=False))

        kernel._send_message(
            "stdout",
            f"Predictions stored in data['{output_name}'] with shape={pred_df.shape}"
        )
