from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

# Responses

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = datetime.now()

class SuccessResponse(BaseModel):
    message: str
    data: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = datetime.now()
