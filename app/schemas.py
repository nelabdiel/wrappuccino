from pydantic import BaseModel
from typing import Optional, List

class PredictionInput(BaseModel):
    # Optional for pipeline mode
    model_name: Optional[str] = None
    preprocessing_module: Optional[str] = None
    use_pipeline: bool = False
    # For text processing tasks
    text: Optional[str] = None
    # For direct numeric input
    features: Optional[list[float]] = None  
