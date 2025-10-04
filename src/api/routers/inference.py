from fastapi import APIRouter, UploadFile, File
from fastapi.responses import Response
import pandas as pd

from src.api.models.responses import SuccessResponse, ErrorResponse
from src.models.toi import TOI
from contextlib import asynccontextmanager
import io


models = {}

@asynccontextmanager
async def lifespan(_: APIRouter):
    """Lifespan of the Inference Router"""
    print(f"Loading models...")
    toi_instance = TOI()
    models["toi"] = toi_instance
    print(f"Model loaded successfully: {toi_instance.model is not None}")
    yield
    print(f"Unloading models...")
    models.clear()


router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    lifespan=lifespan
)

@router.get("/model-status")
async def model_status():
    """Check if the model is properly loaded."""
    toi = models.get("toi")
    if not toi:
        return ErrorResponse(error="Model not initialized", detail="TOI model instance not found")
    
    if not toi.is_loaded():
        return ErrorResponse(error="Model not loaded", detail="TOI model failed to load")
    
    return SuccessResponse(
        message="Model is loaded and ready",
        data=[{
            "model_loaded": True,
            "model_name": getattr(toi, 'model_name', 'Unknown'),
            "model_type": str(type(toi.model)) if toi.model else None
        }]
    )

@router.post("/classify")
async def predict(csv_file: UploadFile = File(...)):
    try:
        csv_data = await csv_file.read()
        df = pd.read_csv(io.BytesIO(csv_data))

    except Exception as e:
        return ErrorResponse(error="Error reading CSV file :(", detail=str(e))
    try:
        toi = models.get("toi")
        if not toi or not toi.is_loaded():
            return ErrorResponse(error="Model not loaded", detail="TOI model failed to load")

        labels = toi.model.predict(df)
        probabilities = toi.model.predict_proba(df)

    except Exception as e:
        return ErrorResponse(error="Error predicting :(", detail=str(e))

    df["label"] = labels
    df["label_confidence"] = [max(probability) for probability in probabilities]
    
    return SuccessResponse(
        message="ITS (NOT) AN EXOPLANET",
        data=df.to_dict(orient="records")
    )

@router.post("/classify-csv")
async def predict_csv(csv_file: UploadFile = File(...)):
    """Classify exoplanets and return results as CSV file download."""
    try:
        csv_data = await csv_file.read()
        df = pd.read_csv(io.BytesIO(csv_data))

    except Exception as e:
        return ErrorResponse(error="Error reading CSV file :(", detail=str(e))
    
    try:
        toi = models.get("toi")
        if not toi or not toi.is_loaded():
            return ErrorResponse(error="Model not loaded", detail="TOI model failed to load")

        labels = toi.model.predict(df)
        probabilities = toi.model.predict_proba(df)

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