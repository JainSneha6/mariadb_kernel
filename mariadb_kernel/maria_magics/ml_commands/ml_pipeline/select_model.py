# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Optional external libraries
_XGBOOST_AVAILABLE = False
_LIGHTGBM_AVAILABLE = False
_CATBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_AVAILABLE = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LIGHTGBM_AVAILABLE = True
except Exception:
    pass

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except Exception:
    pass

class SelectModel(MariaMagic):
    """
    %select_model target=target_col
                  [features=col1,col2] [cv=5] [primary_metric=accuracy|r2|f1|precision|recall|mse|mae]
                  [problem=classification|regression] [output_name=best_model]
                  [inplace=True|False] [model_params={'rf': {'n_estimators': 100}, 'logistic': {'C': 1.0}}]

    Select the best model by comparing all available models on data['last_select'] using cross-validation.
    If features are not provided, uses data['selected_features'] from %select_features.
    Tests all metrics (classification: accuracy, f1, precision, recall; regression: r2, mse, mae).
    Stores the best model in data[output_name] based on primary_metric and displays a table of performances.
    """
    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "select_model"

    def help(self):
        return "Select the best model for training from data['last_select'] using cross-validation."

    def _str_to_obj(self, s):
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
            import json
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

    def _choose_model(self, name, problem, params=None):
        p = params or {}
        name = name.lower()
        if name in ("logistic", "logistic_regression", "lr"):
            if problem != "classification":
                raise ValueError("LogisticRegression is for classification problems.")
            return LogisticRegression(max_iter=1000, **p)
        if name in ("rf", "random_forest"):
            return RandomForestClassifier(**p) if problem == "classification" else RandomForestRegressor(**p)
        if name in ("svc", "svm"):
            if problem != "classification":
                raise ValueError("SVC is for classification problems.")
            return SVC(probability=True, **p)
        if name in ("linear", "linear_regression"):
            if problem != "regression":
                raise ValueError("LinearRegression is for regression problems.")
            return LinearRegression(**p)
        if name == "ridge":
            if problem != "regression":
                raise ValueError("Ridge is for regression problems.")
            return Ridge(**p)
        if name == "lasso":
            if problem != "regression":
                raise ValueError("Lasso is for regression problems.")
            return Lasso(**p)
        if name == "knn":
            return KNeighborsClassifier(**p) if problem == "classification" else KNeighborsRegressor(**p)
        if name == "gbm":
            return GradientBoostingClassifier(**p) if problem == "classification" else GradientBoostingRegressor(**p)
        if name == "ada":
            return AdaBoostClassifier(**p) if problem == "classification" else AdaBoostRegressor(**p)
        if name == "mlp":
            return MLPClassifier(max_iter=1000, **p) if problem == "classification" else MLPRegressor(max_iter=1000, **p)
        if name == "xgboost":
            if not _XGBOOST_AVAILABLE:
                raise ImportError("xgboost not available in this environment.")
            return XGBClassifier(**p) if problem == "classification" else XGBRegressor(**p)
        if name == "lightgbm":
            if not _LIGHTGBM_AVAILABLE:
                raise ImportError("lightgbm not available in this environment.")
            return LGBMClassifier(**p) if problem == "classification" else LGBMRegressor(**p)
        if name == "catboost":
            if not _CATBOOST_AVAILABLE:
                raise ImportError("catboost not available in this environment.")
            p = dict(p)
            p.setdefault("verbose", False)
            return CatBoostClassifier(**p) if problem == "classification" else CatBoostRegressor(**p)
        raise ValueError(f"Unknown model name '{name}'")

    def execute(self, kernel, data):
        # Load training DataFrame
        df = data.get("last_select")
        if df is None or df.empty:
            kernel._send_message("stderr", "No last_select found or DataFrame is empty.")
            return

        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return

        features_arg = args.get("features")
        target = args.get("target")
        cv = int(args.get("cv", 5) or 5)
        primary_metric = args.get("primary_metric", None)
        problem_override = args.get("problem", None)
        output_name = args.get("output_name", "best_model")
        inplace = bool(args.get("inplace", True))
        model_params = args.get("model_params", {}) or {}

        if not target:
            kernel._send_message("stderr", "target argument is required (target=target_col).")
            return

        # Use selected_features if features not provided
        if not features_arg:
            features = data.get("selected_features")
            if not features:
                kernel._send_message("stderr", "No features provided and no selected_features found. Run %select_features first.")
                return
        else:
            if isinstance(features_arg, str):
                features = [c.strip() for c in features_arg.split(",") if c.strip()]
            elif isinstance(features_arg, (list, tuple)):
                features = list(features_arg)
            else:
                kernel._send_message("stderr", "features must be comma-separated string or list.")
                return

        missing = [c for c in features + [target] if c not in df.columns]
        if missing:
            kernel._send_message("stderr", f"Missing columns in DataFrame: {', '.join(missing)}")
            return

        # Determine problem type
        if problem_override:
            problem = problem_override.lower()
            if problem not in ("classification", "regression"):
                kernel._send_message("stderr", "problem must be 'classification' or 'regression'.")
                return
        else:
            tgt_ser = df[target]
            if pd.api.types.is_numeric_dtype(tgt_ser):
                nunique = int(tgt_ser.nunique(dropna=True))
                non_null_count = max(1, len(tgt_ser.dropna()))
                uniq_prop = nunique / non_null_count
                if pd.api.types.is_float_dtype(tgt_ser) or nunique > 20 or uniq_prop > 0.05:
                    problem = "regression"
                else:
                    problem = "classification"
            else:
                problem = "classification"

        # Define all available models based on problem type
        classification_models = ["logistic", "rf", "svm", "knn", "gbm", "ada", "mlp"]
        regression_models = ["linear", "ridge", "lasso", "rf", "knn", "gbm", "ada", "mlp"]
        if _XGBOOST_AVAILABLE:
            classification_models.append("xgboost")
            regression_models.append("xgboost")
        if _LIGHTGBM_AVAILABLE:
            classification_models.append("lightgbm")
            regression_models.append("lightgbm")
        if _CATBOOST_AVAILABLE:
            classification_models.append("catboost")
            regression_models.append("catboost")
        models = classification_models if problem == "classification" else regression_models

        # Define all metrics
        metrics = {
            "classification": ["accuracy", "f1", "precision", "recall"],
            "regression": ["r2", "mse", "mae"]
        }
        if primary_metric is None:
            primary_metric = "accuracy" if problem == "classification" else "r2"
        if primary_metric not in metrics[problem]:
            kernel._send_message("stderr", f"Invalid primary_metric '{primary_metric}' for {problem}. Choose from {', '.join(metrics[problem])}.")
            return

        # Prepare data
        X = df[features].copy()
        y = df[target].copy()

        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True)) if problem == "regression" else X.fillna(X.mode().iloc[0])
        if X.isna().any().any():
            kernel._send_message("stderr", "Features contain non-numeric data or unhandled missing values.")
            return

        # Evaluate models across all metrics
        results = []
        best_model = None
        best_score = -float("inf") if primary_metric not in ("mse", "mae") else float("inf")
        best_model_name = None

        for model_name in models:
            try:
                # Get model-specific parameters
                params = model_params.get(model_name, {}) if isinstance(model_params, dict) else {}
                model = self._choose_model(model_name, problem, params)
                model_result = {"Model": model_name}

                # Evaluate all metrics
                for metric in metrics[problem]:
                    scoring = metric if metric in ("accuracy", "f1", "precision", "recall", "r2") else (
                        "neg_mean_squared_error" if metric == "mse" else "neg_mean_absolute_error"
                    )
                    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    if metric in ("mse", "mae"):
                        mean_score = -mean_score  # Convert to positive
                    model_result[f"{metric}_Mean"] = mean_score
                    model_result[f"{metric}_Std"] = std_score

                results.append(model_result)

                # Update best model based on primary_metric
                current_score = model_result[f"{primary_metric}_Mean"]
                if primary_metric in ("mse", "mae"):
                    if current_score < best_score:
                        best_score = current_score
                        best_model = model
                        best_model_name = model_name
                else:
                    if current_score > best_score:
                        best_score = current_score
                        best_model = model
                        best_model_name = model_name

            except Exception as e:
                kernel._send_message("stderr", f"Error evaluating model '{model_name}': {e}")
                continue

        if not results:
            kernel._send_message("stderr", "No models were successfully evaluated.")
            return

        # Create results DataFrame
        result_df = pd.DataFrame(results)
        for metric in metrics[problem]:
            result_df[f"{metric}_Mean"] = result_df[f"{metric}_Mean"].round(4)
            result_df[f"{metric}_Std"] = result_df[f"{metric}_Std"].round(4)
        result_df = result_df.sort_values(f"{primary_metric}_Mean", ascending=primary_metric in ("mse", "mae"))

        # Fit the best model on the full training data
        try:
            best_model.fit(X, y)
        except Exception as e:
            kernel._send_message("stderr", f"Error fitting best model '{best_model_name}': {e}")
            return

        # Store the best model and metadata
        try:
            data[output_name] = best_model
            data[output_name + "_meta"] = {
                "model_name": best_model_name,
                "problem": problem,
                "features": features,
                "target": target,
                "primary_metric": primary_metric,
                "cv": cv,
                "score": float(best_score),
                "all_results": result_df.to_dict()
            }
            if hasattr(best_model, "classes_"):
                data[output_name + "_meta"]["classes"] = list(getattr(best_model, "classes_"))
        except Exception as e:
            kernel._send_message("stderr", f"Error storing best model: {e}")
            return

        # Display results
        self._send_html(kernel, result_df, title=f"Model Selection Results (primary_metric={primary_metric})")
        kernel._send_message("stdout", f"Best model '{best_model_name}' (mean {primary_metric}={best_score:.4f}) saved to data['{output_name}'].")

        return