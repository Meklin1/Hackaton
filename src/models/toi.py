import joblib
from pathlib import Path
from pandas import DataFrame

MODEL_PATH = Path("models/toi_stacking_pipeline.joblib")

class TOI:
    def __init__(self, model_path: str = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the TOI stacking pipeline model from the specified path."""
        try:
            if self.model_path.exists():
                print(f"Loading model from: {self.model_path}")
                self.model = joblib.load(self.model_path)
                self.model_name = self.model_path.stem
                print(f"Successfully loaded model: {self.model_name}")
                print(f"Model type: {type(self.model)}")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            raise e

    def get_model(self):
        return self.model
    
    def is_loaded(self):
        """Check if the model is properly loaded."""
        return self.model is not None

    def predict(self, df: DataFrame):
        pass

if __name__ == "__main__":

    import pandas as pd

    print("Loading data...")
    path = Path("data/toi_features_only.csv")
    df = pd.read_csv(path)
    print(df.head())

    print("Loading model...")
    toi = TOI()
    print("Predicting...")

    labels = toi.model.predict(df)
    probabilities = toi.model.predict_proba(df)
    
    for label, probability in zip(labels, probabilities):
        print(type(label))
        print(type(probability))
        print(f"Label: {label}, Probability: {probability}")


