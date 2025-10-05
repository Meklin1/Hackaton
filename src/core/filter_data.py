from src.core.train_models import FEATURES
from pandas import DataFrame, to_numeric



def filter_dataset(df: DataFrame) -> DataFrame:

    if "exoplanet_status" in df.columns:
        df = df.drop(columns=["exoplanet_status"])

    
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Prepare data - optimized with direct indexing
    X = df[FEATURES].apply(to_numeric, errors="coerce")

    # Remove rows with all NaN features
    valid_mask = X.notna().any(axis=1)
    X = X[valid_mask].reset_index(drop=True)

    return X