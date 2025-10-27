# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import shlex
from distutils import util
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, mutual_info_classif, mutual_info_regression, chi2, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class SelectFeatures(MariaMagic):
    """
    %select_features target=target_col
                     [method=correlation|rf_importance|rfe|mutual_info|chi2|anova|l1_selection|variance]
                     [k=5] [problem=classification|regression]
                     [output_name=selected_features] [inplace=True|False]

    Identify the best features for training a model on data['last_select'].
    Uses all columns except the target column as features.
    Methods:
    - correlation: Absolute Pearson correlation with the target.
    - rf_importance: RandomForest feature importance scores.
    - rfe: Recursive Feature Elimination with a RandomForest model.
    - mutual_info: Mutual Information between features and target.
    - chi2: Chi-squared statistic (classification only, non-negative features).
    - anova: ANOVA F-test for feature significance.
    - l1_selection: L1-based feature selection (LogisticRegression for classification, Lasso for regression).
    - variance: Remove features with low variance (threshold-based).
    Stores the ranked features in data[output_name] and displays a table of results.
    """
    def __init__(self, args=""):
        self.args = args

    def type(self):
        return "Line"

    def name(self):
        return "select_features"

    def help(self):
        return "Identify the best features for model training from data['last_select']."

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

        target = args.get("target")
        method = args.get("method", "correlation").lower()
        k = args.get("k", 5)
        problem_override = args.get("problem", None)
        output_name = args.get("output_name", "selected_features")
        inplace = bool(args.get("inplace", True))

        if not target:
            kernel._send_message("stderr", "target argument is required (target=target_col).")
            return

        if target not in df.columns:
            kernel._send_message("stderr", f"Target column '{target}' not found in DataFrame.")
            return

        # Use all columns except the target as features
        features = [col for col in df.columns if col != target]
        if not features:
            kernel._send_message("stderr", "No features available after excluding target column.")
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

        # Prepare data
        X = df[features].copy()
        y = df[target].copy()

        # Handle missing values (simple imputation for feature selection)
        X = X.fillna(X.mean(numeric_only=True)) if problem == "regression" else X.fillna(X.mode().iloc[0])
        if X.isna().any().any():
            kernel._send_message("stderr", "Features contain non-numeric data or unhandled missing values.")
            return

        # Scale data for methods that require it
        if method in ("chi2", "l1_selection"):
            scaler = MinMaxScaler() if method == "chi2" else StandardScaler()
            try:
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            except Exception as e:
                kernel._send_message("stderr", f"Error scaling data: {e}")
                return

        # Feature selection
        try:
            if method == "correlation":
                correlations = X.corrwith(y, method="pearson").abs()
                scores = correlations.sort_values(ascending=False)
                selected_features = scores.head(k).index.tolist()
                result_df = pd.DataFrame({
                    "Feature": scores.index,
                    "Score": scores.values
                })

            elif method == "rf_importance":
                model = RandomForestClassifier() if problem == "classification" else RandomForestRegressor()
                model.fit(X, y)
                importances = pd.Series(model.feature_importances_, index=features)
                scores = importances.sort_values(ascending=False)
                selected_features = scores.head(k).index.tolist()
                result_df = pd.DataFrame({
                    "Feature": scores.index,
                    "Score": scores.values
                })

            elif method == "rfe":
                estimator = RandomForestClassifier() if problem == "classification" else RandomForestRegressor()
                selector = RFE(estimator, n_features_to_select=k)
                selector.fit(X, y)
                ranking = pd.Series(selector.ranking_, index=features)
                scores = 1 / (ranking + 1)
                selected_features = ranking[ranking == 1].index.tolist()
                result_df = pd.DataFrame({
                    "Feature": ranking.index,
                    "Score": scores,
                    "Ranking": ranking
                }).sort_values("Score", ascending=False)

            elif method == "mutual_info":
                score_func = mutual_info_classif if problem == "classification" else mutual_info_regression
                selector = SelectKBest(score_func=score_func, k=k)
                selector.fit(X, y)
                scores = pd.Series(selector.scores_, index=features)
                scores = scores.sort_values(ascending=False)
                selected_features = scores.head(k).index.tolist()
                result_df = pd.DataFrame({
                    "Feature": scores.index,
                    "Score": scores.values
                })

            elif method == "chi2":
                if problem != "classification":
                    kernel._send_message("stderr", "chi2 method is only for classification problems.")
                    return
                if (X < 0).any().any():
                    kernel._send_message("stderr", "chi2 requires non-negative features.")
                    return
                selector = SelectKBest(score_func=chi2, k=k)
                selector.fit(X, y)
                scores = pd.Series(selector.scores_, index=features)
                scores = scores.sort_values(ascending=False)
                selected_features = scores.head(k).index.tolist()
                result_df = pd.DataFrame({
                    "Feature": scores.index,
                    "Score": scores.values
                })

            elif method == "anova":
                score_func = f_classif if problem == "classification" else f_regression
                selector = SelectKBest(score_func=score_func, k=k)
                selector.fit(X, y)
                scores = pd.Series(selector.scores_, index=features)
                scores = scores.sort_values(ascending=False)
                selected_features = scores.head(k).index.tolist()
                result_df = pd.DataFrame({
                    "Feature": scores.index,
                    "Score": scores.values
                })

            elif method == "l1_selection":
                model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000) if problem == "classification" else Lasso(alpha=0.01)
                model.fit(X, y)
                scores = pd.Series(np.abs(model.coef_.ravel() if problem == "classification" else model.coef_), index=features)
                scores = scores.sort_values(ascending=False)
                selected_features = scores[scores > 0].head(k).index.tolist()
                result_df = pd.DataFrame({
                    "Feature": scores.index,
                    "Score": scores.values
                })

            elif method == "variance":
                selector = VarianceThreshold(threshold=0.0)
                selector.fit(X)
                variances = pd.Series(selector.variances_, index=features)
                scores = variances.sort_values(ascending=False)
                selected_features = scores.head(k).index.tolist()
                result_df = pd.DataFrame({
                    "Feature": scores.index,
                    "Score": scores.values
                })

            else:
                kernel._send_message("stderr", "method must be one of 'correlation', 'rf_importance', 'rfe', 'mutual_info', 'chi2', 'anova', 'l1_selection', or 'variance'.")
                return

        except Exception as e:
            kernel._send_message("stderr", f"Error during feature selection: {e}")
            return

        # Store results
        try:
            data[output_name] = selected_features
            data[output_name + "_meta"] = {
                "method": method,
                "problem": problem,
                "target": target,
                "k": k,
                "all_scores": result_df.to_dict()
            }
        except Exception as e:
            kernel._send_message("stderr", f"Error storing results: {e}")
            return

        # Display results
        self._send_html(kernel, result_df, title=f"Feature Selection Results (method={method})")
        kernel._send_message("stdout", f"Selected {len(selected_features)} features saved to data['{output_name}']: {', '.join(selected_features)}")

        return