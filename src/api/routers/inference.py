from fastapi import APIRouter, UploadFile, File
import pandas as pd
import io
from src.api.models.responses import SuccessResponse, ErrorResponse


router = APIRouter(
    prefix="/inference",
    tags=["inference"]
)

@router.post("/classify")
async def predict(csv_file: UploadFile = File(...)):
    try:
        csv_data = await csv_file.read()
        df = pd.read_csv(io.BytesIO(csv_data))
    except Exception as e:
        return ErrorResponse(error="Error reading CSV file :(", detail=str(e))
    return SuccessResponse(
        message="ITS (NOT) AN EXOPLANET",
        data=df.to_dict(orient="records")
    )