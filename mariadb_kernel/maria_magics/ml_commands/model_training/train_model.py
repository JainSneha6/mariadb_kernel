# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

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


class TrainModel(MariaMagic):
    """
    %train_model model=<name> features=col1,col2 target=target_col
                 [cv=0] [problem=classification|regression]
                 [model_name=last_model] [pred_name=last_preds] [test_name=last_select_test]
                 [save_path=/path/to/model.joblib] [inplace=True|False] [model_params={'n':1}]

    Train a model on data["last_select"] (TRAINING set). This magic DOES NOT perform
    splitting or scaling — run your preprocessing and %splitdata beforehand.
    """
    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "train_model"

    def help(self):
        return "Train a model on data['last_select'] (no split or scaling)."

    def _str_to_obj(self, s):
        # try int/float/bool, then JSON, then string unquote
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
        # try json
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
        # Classification vs regression models where appropriate
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
            # CatBoost often prints to stdout; keep default verbose False
            p = dict(p)
            p.setdefault("verbose", False)
            return CatBoostClassifier(**p) if problem == "classification" else CatBoostRegressor(**p)
        raise ValueError(f"Unknown model name '{name}'")

    def execute(self, kernel, data):
        # Load training DataFrame
        df = data.get("last_select")
        if df is None or df.empty:
            kernel._send_message("stderr", "No last_select found or DataFrame is empty (training set required).")
            return

        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments. Use key=value syntax.")
            return

        features_arg = args.get("features")
        target = args.get("target")
        model_name_arg = args.get("model", "rf")
        cv = int(args.get("cv", 0) or 0)
        problem_override = args.get("problem", None)
        test_name = args.get("test_name", "last_select_test")
        model_store_name = args.get("model_name", "last_model")
        # pred_name and save_path intentionally ignored/removed
        inplace = bool(args.get("inplace", True))
        model_params = args.get("model_params", {}) or {}

        if not features_arg:
            kernel._send_message("stderr", "features argument is required (features=col1,col2...).")
            return
        if not target:
            kernel._send_message("stderr", "target argument is required (target=target_col).")
            return

        # parse features
        if isinstance(features_arg, str):
            features = [c.strip() for c in features_arg.split(",") if c.strip()]
        elif isinstance(features_arg, (list, tuple)):
            features = list(features_arg)
        else:
            kernel._send_message("stderr", "features must be comma-separated string or list.")
            return

        missing = [c for c in features + [target] if c not in df.columns]
        if missing:
            kernel._send_message("stderr", f"Missing columns in training DataFrame: {', '.join(missing)}")
            return

        # Determine problem type
        if problem_override:
            problem = problem_override.lower()
            if problem not in ("classification", "regression"):
                kernel._send_message("stderr", "problem must be 'classification' or 'regression'.")
                return
        else:
            # improved heuristic for problem detection
            tgt_ser = df[target]

            if pd.api.types.is_numeric_dtype(tgt_ser):
                nunique = int(tgt_ser.nunique(dropna=True))
                non_null_count = max(1, len(tgt_ser.dropna()))
                uniq_prop = nunique / non_null_count

                # treat as regression if:
                #  - float dtype, or
                #  - many distinct values (>20), or
                #  - distinct proportion high (e.g. >5% of rows)
                if pd.api.types.is_float_dtype(tgt_ser) or (nunique > 20) or (uniq_prop > 0.05):
                    problem = "regression"
                else:
                    # few distinct integer-like values -> classification (categorical target)
                    problem = "classification"
            else:
                problem = "classification"

        # Prepare X_train, y_train
        X_train = df[features].copy()
        y_train = df[target].copy()

        # NOTE: test set (if present) will be ignored in this modified flow — no predictions or metrics.
        # Keep reading test_df only to validate presence but do not use it.
        test_df = data.get(test_name)
        if isinstance(test_df, pd.DataFrame) and not test_df.empty:
            missing_test = [c for c in features + [target] if c not in test_df.columns]
            if missing_test:
                kernel._send_message("stderr", f"Test DataFrame '{test_name}' missing columns: {', '.join(missing_test)}")
                return

        # Instantiate model
        try:
            model = self._choose_model(model_name_arg, problem, params=model_params)
        except Exception as e:
            kernel._send_message("stderr", f"Error creating model: {e}")
            return

        # Cross-validation on training set if requested (kept)
        cv_results = None
        if cv and cv > 1:
            try:
                scoring = "accuracy" if problem == "classification" else "r2"
                cv_results = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            except Exception as e:
                kernel._send_message("stderr", f"Error during cross-validation: {e}")
                return

        # Fit
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            kernel._send_message("stderr", f"Error fitting model: {e}")
            return

        # Store only the trained model and minimal meta (no preds, no metrics, no joblib saving)
        try:
            data[model_store_name] = model

            # Save metadata including target so evaluate_model can find it
            meta = data.setdefault(model_store_name + "_meta", {})
            meta["problem"] = problem
            meta["features"] = features
            meta["target"] = target

            # If model exposes classes_, save them for easier decoding later
            if hasattr(model, "classes_"):
                try:
                    meta["classes"] = list(getattr(model, "classes_"))
                except Exception:
                    pass

        except Exception as e:
            kernel._send_message("stderr", f"Error storing model: {e}")
            return

        # Output concise summary
        out_lines = [f"Model '{model_name_arg}' trained and saved to data['{model_store_name}']. problem={problem}. train_rows={len(X_train)}"]
        if cv_results is not None:
            out_lines.append(f"cross-val (cv={cv}) scores: mean={float(np.mean(cv_results)):.4f}, std={float(np.std(cv_results)):.4f}")
        kernel._send_message("stdout", "\n".join(out_lines))

        return
