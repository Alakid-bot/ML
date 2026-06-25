from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class AdaptiveLoadPredictor(RegressorMixin, BaseEstimator):
    """Ridge backbone with optional MLP residual correction."""

    def __init__(
        self,
        *,
        ridge_alpha: float = 1.0,
        residual_weight: float = 0.75,
        min_samples_for_residual: int = 50,
        validation_size: float = 0.2,
        min_improvement: float = 0.01,
        mlp_hidden_layer_sizes: tuple[int, ...] = (64, 32),
        mlp_alpha: float = 1e-4,
        mlp_max_iter: int = 1000,
        random_state: int | None = 42,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.residual_weight = residual_weight
        self.min_samples_for_residual = min_samples_for_residual
        self.validation_size = validation_size
        self.min_improvement = min_improvement
        self.mlp_hidden_layer_sizes = mlp_hidden_layer_sizes
        self.mlp_alpha = mlp_alpha
        self.mlp_max_iter = mlp_max_iter
        self.random_state = random_state

    def fit(self, X: object, y: object) -> "AdaptiveLoadPredictor":
        X_checked, y_checked = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X_checked.shape[1]

        self.backbone_ = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        self.residual_model_ = None
        self.residual_enabled_ = False
        self.backbone_rmse_ = None
        self.hybrid_rmse_ = None

        if len(X_checked) < self.min_samples_for_residual:
            self.backbone_.fit(X_checked, y_checked)
            self.selected_strategy_ = "ridge_backbone_small_dataset"
            return self

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_checked,
            y_checked,
            test_size=self.validation_size,
            random_state=self.random_state,
        )

        self.backbone_.fit(X_train, y_train)
        train_backbone_pred = self.backbone_.predict(X_train)
        residual_target = y_train - train_backbone_pred

        candidate_residual = MLPRegressor(
            hidden_layer_sizes=self.mlp_hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.mlp_alpha,
            early_stopping=True,
            max_iter=self.mlp_max_iter,
            random_state=self.random_state,
        )
        candidate_residual.fit(X_train, residual_target)

        backbone_valid_pred = self.backbone_.predict(X_valid)
        residual_valid_pred = candidate_residual.predict(X_valid)
        hybrid_valid_pred = backbone_valid_pred + self.residual_weight * residual_valid_pred

        backbone_rmse = float(root_mean_squared_error(y_valid, backbone_valid_pred))
        hybrid_rmse = float(root_mean_squared_error(y_valid, hybrid_valid_pred))
        self.backbone_rmse_ = backbone_rmse
        self.hybrid_rmse_ = hybrid_rmse

        improvement = (backbone_rmse - hybrid_rmse) / backbone_rmse if backbone_rmse > 0 else 0.0
        self.residual_enabled_ = improvement >= self.min_improvement

        self.backbone_.fit(X_checked, y_checked)
        if self.residual_enabled_:
            full_residual_target = y_checked - self.backbone_.predict(X_checked)
            self.residual_model_ = MLPRegressor(
                hidden_layer_sizes=self.mlp_hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.mlp_alpha,
                early_stopping=True,
                max_iter=self.mlp_max_iter,
                random_state=self.random_state,
            )
            self.residual_model_.fit(X_checked, full_residual_target)
            self.selected_strategy_ = "ridge_plus_mlp_residual"
        else:
            self.selected_strategy_ = "ridge_backbone_validation_fallback"

        return self

    def predict(self, X: object) -> np.ndarray:
        check_is_fitted(self, "backbone_")
        X_checked = check_array(X)
        prediction = self.backbone_.predict(X_checked)

        if self.residual_enabled_ and self.residual_model_ is not None:
            prediction = prediction + self.residual_weight * self.residual_model_.predict(X_checked)

        return np.asarray(prediction)

    def model_metadata(self) -> dict[str, object]:
        check_is_fitted(self, "backbone_")
        return {
            "model_type": type(self).__name__,
            "strategy": self.selected_strategy_,
            "residual_enabled": self.residual_enabled_,
            "residual_weight": self.residual_weight,
            "min_samples_for_residual": self.min_samples_for_residual,
            "backbone_model": "Ridge",
            "residual_model": "MLPRegressor",
            "backbone_rmse": self.backbone_rmse_,
            "hybrid_rmse": self.hybrid_rmse_,
        }
