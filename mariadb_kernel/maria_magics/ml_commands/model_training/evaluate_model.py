# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, classification_report
)
from sklearn.preprocessing import LabelEncoder

class EvaluateModel(MariaMagic):
    """
    %evaluate_model [model_name=last_model] [test_name=last_select_test] [pred_name=last_preds]
                    [problem=classification|regression]

    Evaluate a previously trained model stored in `data[model_name]` using the test
    DataFrame `data[test_name]`. Outputs metrics, confusion matrix, ROC AUC (if applicable),
    and displays a table with actual vs predicted (and probabilities if available).
    """
    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "evaluate_model"

    def help(self):
        return "Evaluate a trained model on a test DataFrame and show metrics + predictions."

    def _str_to_obj(self, s):
        # same helper as in TrainModel
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
        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return

        model_store_name = args.get("model_name", args.get("model", "last_model"))
        test_name = args.get("test_name", "last_select_test")
        pred_name = args.get("pred_name", "last_preds")
        problem_override = args.get("problem", None)

        # fetch model
        model = data.get(model_store_name)
        if model is None:
            kernel._send_message("stderr", f"No model found in data['{model_store_name}']. Train and save a model first.")
            return

        # fetch test set
        test_df = data.get(test_name)
        if test_df is None or not isinstance(test_df, pd.DataFrame) or test_df.empty:
            kernel._send_message("stderr", f"No test DataFrame found in data['{test_name}'] or it is empty.")
            return

        # try to infer problem from model type if not provided
        if problem_override:
            problem = problem_override.lower()
        else:
            is_classifier = any(attr in dir(model) for attr in ("predict_proba", "decision_function", "classes_"))
            problem = "classification" if is_classifier else "regression"

        # get meta (features + target) from training metadata
        meta = data.get(model_store_name + "_meta", {}) or {}
        features = meta.get("features")
        target_col = meta.get("target") or meta.get("target_col")

        # fallback: if no target in meta, try to infer target as the only non-feature column
        if not target_col:
            if features:
                possible_targets = [c for c in test_df.columns if c not in features]
                if len(possible_targets) == 1:
                    target_col = possible_targets[0]

        if not target_col:
            kernel._send_message("stderr", "Target column not found in model meta and could not be inferred from test DataFrame. "
                                         "Set data[model_name + '_meta']['target']='<target_column>' when training, or pass target info in meta.")
            return

        if target_col not in test_df.columns:
            kernel._send_message("stderr", f"Target column '{target_col}' not present in test DataFrame '{test_name}'.")
            return

        if not features:
            kernel._send_message("stderr", "Model metadata does not contain 'features' list. Cannot build X_test.")
            return

        missing_features = [c for c in features if c not in test_df.columns]
        if missing_features:
            kernel._send_message("stderr", f"Test DataFrame missing feature columns: {', '.join(missing_features)}")
            return

        X_test = test_df[features].copy()
        y_true_orig = test_df[target_col].copy()  # preserve original values for display

        # Predict
        try:
            preds_raw = model.predict(X_test)
        except Exception as e:
            kernel._send_message("stderr", f"Error during prediction: {e}")
            return

        # Try predict_proba
        pred_proba = None
        if problem == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    pred_proba = proba[:, 1].tolist()
                else:
                    pred_proba = proba.tolist()
            except Exception:
                pred_proba = None

        # Try to build preds_display (human-readable)
        preds_display = preds_raw
        model_classes = getattr(model, "classes_", None)
        try:
            # if model has classes_ and preds_raw are indices, map to class labels
            if model_classes is not None and pd.api.types.is_integer_dtype(np.asarray(preds_raw).dtype):
                preds_display = np.asarray(model_classes)[np.asarray(preds_raw).astype(int)]
            # if preds_raw are numeric but classes_ are strings, try mapping by index
            elif model_classes is not None and not pd.api.types.is_numeric_dtype(model_classes):
                # if preds_raw are label indices (ints) handle above; else if preds_raw are strings keep as-is
                pass
        except Exception:
            preds_display = preds_raw

        # Construct predictions DataFrame
        preds_df = test_df.copy(deep=True)
        preds_df["_predicted"] = preds_display
        if pred_proba is not None:
            preds_df["_pred_proba"] = pred_proba

        data[pred_name] = preds_df

        # --- Metrics: ensure consistent types for y_true and preds ---
        out_lines = []
        if problem == "classification":
            # build arrays for metrics
            y_true_vals = np.asarray(y_true_orig)
            preds_vals = np.asarray(preds_display)

            # If types are mixed (numbers and strings), cast both to str for label-based metrics
            def is_mixed(a, b):
                return (pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_numeric_dtype(b)) or \
                       (pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_numeric_dtype(a))

            if is_mixed(y_true_vals.dtype, preds_vals.dtype):
                y_metric = np.asarray(y_true_orig.astype(str))
                p_metric = np.asarray(pd.Series(preds_display).astype(str))
            else:
                # prefer numeric if both numeric; else use original dtype (strings)
                if pd.api.types.is_numeric_dtype(y_true_vals) and pd.api.types.is_numeric_dtype(preds_vals):
                    y_metric = y_true_vals.astype(float)
                    p_metric = preds_vals.astype(float)
                else:
                    y_metric = np.asarray(y_true_orig.astype(str))
                    p_metric = np.asarray(pd.Series(preds_display).astype(str))

            # compute basic metrics (these accept string labels fine)
            try:
                acc = accuracy_score(y_metric, p_metric)
                prec = precision_score(y_metric, p_metric, average="weighted", zero_division=0)
                rec = recall_score(y_metric, p_metric, average="weighted", zero_division=0)
                f1 = f1_score(y_metric, p_metric, average="weighted", zero_division=0)
                cm = confusion_matrix(y_metric, p_metric)
            except Exception as e:
                kernel._send_message("stderr", f"Error computing classification metrics: {e}")
                return

            out_lines.append(f"Classification metrics (model: '{model_store_name}')")
            out_lines.append(f"  accuracy = {acc:.4f}")
            out_lines.append(f"  precision (weighted) = {prec:.4f}")
            out_lines.append(f"  recall (weighted) = {rec:.4f}")
            out_lines.append(f"  f1 (weighted) = {f1:.4f}")
            out_lines.append("  Confusion matrix (rows=actual, cols=predicted):")
            out_lines.append(str(cm.tolist()))

            # ROC AUC: attempt only if predict_proba available and we can map y_true to integer indices
            roc_text = "  ROC AUC not available."
            if pred_proba is not None and model_classes is not None:
                try:
                    # map true labels to indices using model.classes_
                    class_to_idx = {str(c): i for i, c in enumerate(model_classes)}
                    y_idx = np.array([class_to_idx.get(str(v), None) for v in y_true_orig])
                    if None in y_idx:
                        roc_text = "  ROC AUC not computable: some test classes not present in model.classes_."
                    else:
                        proba_arr = np.asarray(pred_proba)
                        if proba_arr.ndim == 1:
                            # binary case
                            roc_auc = roc_auc_score(y_idx.astype(int), proba_arr.astype(float))
                            roc_text = f"  ROC AUC (binary) = {roc_auc:.4f}"
                        else:
                            roc_auc = roc_auc_score(y_idx.astype(int), proba_arr, multi_class="ovr", average="weighted")
                            roc_text = f"  ROC AUC (multiclass OVR, weighted) = {roc_auc:.4f}"
                except Exception:
                    roc_text = "  ROC AUC computation failed."
            elif pred_proba is not None and model_classes is None:
                roc_text = "  ROC AUC not computed: model.classes_ missing."
            out_lines.append(roc_text)

            # classification report with readable labels if possible
            try:
                target_names = None
                if model_classes is not None:
                    target_names = [str(c) for c in model_classes]
                report = classification_report(y_metric, p_metric, zero_division=0, target_names=target_names)
                out_lines.append("\nClassification report:\n" + report)
            except Exception:
                pass

        else:
            # regression metrics
            try:
                preds_num = np.asarray(preds_raw).astype(float)
                y_true_num = np.asarray(y_true_orig).astype(float)
                rmse = float(np.sqrt(mean_squared_error(y_true_num, preds_num)))
                mae = float(mean_absolute_error(y_true_num, preds_num))
                r2 = float(r2_score(y_true_num, preds_num))
                out_lines.append(f"Regression metrics (model: '{model_store_name}')")
                out_lines.append(f"  RMSE = {rmse:.4f}")
                out_lines.append(f"  MAE  = {mae:.4f}")
                out_lines.append(f"  R2   = {r2:.4f}")
            except Exception as e:
                kernel._send_message("stderr", f"Error computing regression metrics: {e}")
                return

        # send textual summary
        kernel._send_message("stdout", "\n".join(out_lines))

        # display HTML preview of predictions (actual vs predicted)
        try:
            self._send_html(kernel, preds_df.head(200),
                            title=f"Predictions (actual={target_col} | predicted=_predicted). Showing up to 200 rows.")
        except Exception:
            pass

        return
