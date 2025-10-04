from pydantic import BaseModel
from typing import Optional, Dict, Any

# Requests

class TrainModelRequest(BaseModel):
    model_name: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None

