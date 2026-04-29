from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def get_candidate_models():

    models = {

        # ---------------------------------------------------
        # 1. Logistic Regression (PRIMARY MODEL)
        # ---------------------------------------------------
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        # ---------------------------------------------------
        # 2. XGBoost (performance comparison)
        # ---------------------------------------------------
        "xgboost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=3,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
    }

    return models