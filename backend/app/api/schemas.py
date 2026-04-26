from pydantic import BaseModel


class IssuesSchema(BaseModel):
    blurry: int
    low_light: int
    cluttered: int
    bad_framing: int
    tilted: int
    poor_space_visibility: int
    watermark: int


class PredictResponse(BaseModel):
    issues: IssuesSchema
    score: int
    grade: str
    suggestions: str
