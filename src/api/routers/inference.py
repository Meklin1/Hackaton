from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import Response
import pandas as pd

from src.api.models.responses import SuccessResponse, ErrorResponse, ModelDataResponse
from src.models.toi import TOI
from contextlib import asynccontextmanager
from pathlib import Path 
import os
import io


loaded_models = {}

def get_models_info() -> dict:
    """
    Returns a dictionary of available models in the 'models' directory.
    Format:
    {
        "0": {
            "name": <model_name_without_extension>,
            "path": <Path object to model file>
        },
        ...
    }
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return {}
    
    try:
        models = [
            model for model in os.listdir(models_dir)
            if model.endswith(".joblib")
        ]
        return {
            str(idx): {
                "name": os.path.splitext(model)[0],
                "path": Path(models_dir) / model
            }
            for idx, model in enumerate(models)
        }
    except Exception as e:
        print(f"Error reading models directory: {e}")
        return {}

@asynccontextmanager
async def lifespan(_: APIRouter):
    """Lifespan of the Inference Router"""
    print(f"Loading models...")
    
    try:
        for model in os.listdir("models"):
            model_name = model.split(".")[0]
            loaded_models[model_name] = TOI(Path("models", f"{model}"))
            print(f"Model loaded successfully: {loaded_models[model_name].is_loaded()}")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e
    
    yield
    print(f"Unloading models...")
    loaded_models.clear()


router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    lifespan=lifespan
)


@router.get("/get-models")
async def get_models(models_info: dict = Depends(get_models_info)): 
    return ModelDataResponse(data=models_info)


@router.post("/classify")
async def predict(csv_file: UploadFile = File(...), model_id: str = "0", models_info: dict = Depends(get_models_info)):
    try:
        csv_data = await csv_file.read()
        df = pd.read_csv(io.BytesIO(csv_data))

    except Exception as e:
        return ErrorResponse(error="Error reading CSV file :(", detail=str(e))
    try:
        model_name = models_info.get(model_id).get("name")
        inference_model = loaded_models.get(model_name)
        if not inference_model or not inference_model.is_loaded():
            return ErrorResponse(error="Model not loaded", detail="TOI model failed to load")

        labels = inference_model.model.predict(df)
        probabilities = inference_model.model.predict_proba(df)

    except Exception as e:
        return ErrorResponse(error="Error predicting :(", detail=str(e))

    df["label"] = labels
    df["label_confidence"] = [max(probability) for probability in probabilities]
    
    return SuccessResponse(
        message="ITS (NOT) AN EXOPLANET",
        data=df.to_dict(orient="records")
    )

@router.post("/classify-csv")
async def predict_csv(csv_file: UploadFile = File(...), model_id: str = "0", models_info: dict = Depends(get_models_info)):
    """Classify exoplanets and return results as CSV file download."""
    try:
        csv_data = await csv_file.read()
        df = pd.read_csv(io.BytesIO(csv_data))

    except Exception as e:
        return ErrorResponse(error="Error reading CSV file :(", detail=str(e))
    
    try:
        model_name = models_info.get(model_id).get("name")
        inference_model = loaded_models.get(model_name)
        if not inference_model or not inference_model.is_loaded():
            return ErrorResponse(error="Model not loaded", detail="TOI model failed to load")

        labels = inference_model.model.predict(df)
        probabilities = inference_model.model.predict_proba(df)
    except Exception as e:
        return ErrorResponse(error="Error predicting :(", detail=str(e))

    # Add predictions to the dataframe
    df["label"] = labels
    df["label_confidence"] = [max(probability) for probability in probabilities]
    
    # Convert to CSV
    csv_output = df.to_csv(index=False)
    
    # Return as CSV file download
    return Response(
        content=csv_output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=exoplanet_predictions.csv"}
    )