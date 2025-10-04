from fastapi import APIRouter, UploadFile, File, Form
import pandas as pd
from src.core.train_models import train_exoplanet_model
from src.api.models.requests import TrainModelRequest
from src.api.models.responses import SuccessResponse, ErrorResponse
from typing import Optional, Dict, Any
import uuid
import os
import io
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


router = APIRouter(
    prefix="/model",
    tags=["model"],
)

@router.post("/train")
async def train_model(
        csv_file: UploadFile = File(...), 
        model_name: str = Form(default_factory=lambda: str(uuid.uuid4())),
        hyperparameters: str = Form(default="{}")
        ):
    try:
        csv_data = await csv_file.read()
        df = pd.read_csv(io.BytesIO(csv_data))

    except Exception as e:
        return ErrorResponse(error="Error reading CSV file :(", detail=str(e))

    try:
        # Parse hyperparameters JSON string
        hyperparams_dict = json.loads(hyperparameters) if hyperparameters else {}
    except json.JSONDecodeError as e:
        return ErrorResponse(error="Invalid hyperparameters JSON format :(", detail=str(e))

    try:
        logger.info("Training Started")
        traing_status = await train_exoplanet_model(df, model_name, hyperparams_dict, output_path="models")
        logger.info("Finish training")
    except Exception as e:
        return ErrorResponse(error=f"Error training model :( {str(e)}", detail=str(e))

   
    return SuccessResponse(
        message="Model trained successfully :D!",
        data=[traing_status]
        )
