from pydantic import BaseModel

class SynergiVerificationRequest(BaseModel):
    year: int
    month: str
    location: str
    substance: str

class SynergiVerificationValue(BaseModel):
    year: int
    month: str
    location: str
    substance: str
    amount: float
