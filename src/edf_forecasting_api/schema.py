from pydantic import BaseModel
from typing import List, Optional

class InputData(BaseModel):
    features: List[List[float]]
    n_predictions: Optional[int] = 1