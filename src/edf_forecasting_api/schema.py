from pydantic import BaseModel
from typing import List, Optional

class InputData(BaseModel):
    features: List[List[float]]
    n_predictions: Optional[int] = 1

class FeedbackData(BaseModel):
    prediction_id: str
    inputs: List[List[float]]
    true_values: List[List[float]]