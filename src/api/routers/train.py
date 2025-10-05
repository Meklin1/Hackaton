from fastapi import APIRouter, Form, Depends
import pandas as pd

from src.core.train_models import train_exoplanet_model
from src.api.models.responses import SuccessResponse, ErrorResponse
from src.api.routers.inference import loaded_models, load_model_dynamically
from src.api.routers.dataset import get_dataset_info

from pathlib import Path
import uuid
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
    csv_id: str = None,
    model_name: str = Form(default_factory=lambda: str(uuid.uuid4())),
    hyperparameters: str = Form(default="{}"),
    datasets_info: dict = Depends(get_dataset_info),
):
    try:
        csv_path = datasets_info.get(csv_id).get("path")
        df = pd.read_csv(Path(csv_path))

    except Exception as e:
        return ErrorResponse(error="Error reading CSV file :(", detail=str(e))

    try:
        # Parse hyperparameters JSON string
        logger.info(f"Hyperparameters: {hyperparameters}")
        hyperparams_dict = json.loads(hyperparameters) if hyperparameters else {}
        logger.info(f"Decoded Hyperparameters: {hyperparams_dict}")
    except json.JSONDecodeError as e:
        return ErrorResponse(
            error="Invalid hyperparameters JSON format :(", detail=str(e)
        )

    try:
        logger.info("Training Started")
        traing_status = await train_exoplanet_model(
            df, model_name, hyperparams_dict, output_path="models"
        )
        logger.info("Finish training")
        
        # Dynamically load the newly trained model
        model_path = Path(traing_status["model_path"])
        model_name_clean = model_name.replace(".joblib", "")
        
        logger.info(f"Loading new model dynamically: {model_name_clean}")
        load_success = load_model_dynamically(model_name_clean, model_path)
        
        if not load_success:
            logger.warning(f"Failed to load model {model_name_clean} dynamically")
            # Don't fail the training, just warn - model will be available after restart
        
    except Exception as e:
        return ErrorResponse(error=f"Error training model :( {str(e)}", detail=str(e))

    return SuccessResponse(
        message=f"Model trained successfully :D! in {len(df)}", data=[traing_status]
    )
