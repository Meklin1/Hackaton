import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from lightgbm import LGBMClassifier
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES = [
    'pl_orbper', 'pl_trandurh', 'pl_rade', 'st_dist', 'st_pmdec',
    'st_pmra', 'dec', 'pl_insol', 'pl_tranmid', 'ra', 'st_tmag',
    'pl_trandep', 'pl_eqt', 'st_rad', 'st_logg', 'st_teff'
]

# Optimized hyperparameters - smaller search space for faster training
DEFAULT_PARAMS = {
    'lgbm': {
        'n_estimators': [200, 400],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [5, 10],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    },
    'gb': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
}


async def train_exoplanet_model(
    df: pd.DataFrame,
    model_name: str,
    hyperparameters: dict = None,
    output_path: str = "models"
    
) -> dict:
    """
    Train exoplanet classification model and save to disk.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and 'exoplanet_status' column (1 or 0)
    model_name : str
        Name for the output model file
    hyperparameters : dict, optional
        Custom hyperparameter grids for 'lgbm' and 'gb' models
    output_path : str, optional
        Directory path to save model. Defaults to current directory

    Returns:
    --------
    dict with keys:
        - 'model_path': Path to saved model file
        - 'accuracy': Test accuracy
        - 'roc_auc': Test ROC AUC score
        - 'f1_score': Test F1 score
    """
    # Validate input
    if 'exoplanet_status' not in df.columns:
        raise ValueError("DataFrame must contain 'exoplanet_status' column")

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Prepare data - optimized with direct indexing
    X = df[FEATURES].apply(pd.to_numeric, errors='coerce')
    y = df['exoplanet_status'].astype(int)

    # Remove rows with all NaN features
    valid_mask = ~X.isna().all(axis=1)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    if len(y.unique()) < 2:
        raise ValueError("Dataset must contain both classes (0 and 1)")

    # Single split instead of two splits - faster
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y
    )

    # Preprocessing
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    # Use custom or default hyperparameters
    params = hyperparameters if hyperparameters else DEFAULT_PARAMS

    # Optimized: Reduced n_iter and cv folds for faster search
    # Tune LightGBM
    lgb = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    rs_lgb = RandomizedSearchCV(
        lgb, param_distributions=params.get('lgbm', DEFAULT_PARAMS['lgbm']),
        n_iter=10,  # Reduced from 20
        scoring='roc_auc',
        cv=3,  # Reduced from 5
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )
    logger.info("Tuning LightGBM")
    rs_lgb.fit(X_train_p, y_train)
    best_lgb = rs_lgb.best_estimator_

    # Tune Gradient Boosting
    gb = GradientBoostingClassifier(random_state=RANDOM_SEED)
    rs_gb = RandomizedSearchCV(
        gb, param_distributions=params.get('gb', DEFAULT_PARAMS['gb']),
        n_iter=8,  # Reduced from 15
        scoring='roc_auc',
        cv=3,  # Reduced from 5
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0
    )

    logger.info("Tuning Gradient Boosting")
    rs_gb.fit(X_train_p, y_train)
    best_gb = rs_gb.best_estimator_

    # Train stacking ensemble with reduced CV folds
    stack = StackingClassifier(
        estimators=[('lgbm', best_lgb), ('gb', best_gb)],
        final_estimator=LogisticRegression(max_iter=500, n_jobs=-1),
        n_jobs=-1,
        cv=3  # Reduced from 5
    )
    logger.info("Training Stacking Ensemble")
    stack.fit(X_train_p, y_train)

    # Create final pipeline
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', stack)
    ])

    # Evaluate on test set 
    logger.info("Evaluating on test set")
    y_test_pred = final_pipeline.predict(X_test)
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'f1_score': f1_score(y_test, y_test_pred)
    }

    # Save model
    output_dir = Path(output_path) if output_path else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_name.endswith('.joblib'):
        model_name = f"{model_name}.joblib"

    model_path = output_dir / model_name
    joblib.dump(final_pipeline, model_path)

    return {
        'model_path': str(model_path),
        **metrics
    }