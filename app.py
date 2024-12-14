from fastapi import FastAPI
from domain.domain import SynergiVerificationRequest, SynergiVerificationValue
from service.SynergiService import SynergiService

synergi_value_app = FastAPI()

@synergi_value_app.post("/predict")
async def prdict_value(request: SynergiVerificationRequest)->SynergiVerificationValue:
    return SynergiService().predict_value(request=request)