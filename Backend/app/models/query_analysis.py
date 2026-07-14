from pydantic import BaseModel,Field
from typing import Literal

class QeryAnalysis(BaseModel):
    classification: Literal["CONCEPTUAL","CODE_SPECIFIC"] = Field(
        description=(
            "CONCEPTUAL for 'how/why does X work' style questions.",
            "CODE_SPECIFIC for 'where is X defined/called/implemented' style questions."
        )              
    )

    confidence:float = Field(ge=0.0,le=1.0,description="Confidence in classification,0 to 1")
    
    expanded_queries:list[str] = Field(
        min_length=1,
        max_length=3,
        description=(
            "2-3 alternative search queries using real function/class/file",
            "names or technical synonyms visible in the provided context"
        )
    )