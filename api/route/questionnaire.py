import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from api.service import chat_handler

router = APIRouter()
prefix = "/questionnaire"

class Input(BaseModel):
    description: str

file_map = {
    "Biofuels": "biofuels.json",
    "Biotechnology & Pharmaceuticals": "biotech.json",
    "Software & IT Services": "software-and-it.json",
    "Food Retailers & Distributors": "food-retail.json",
    "Oil & Gas – Exploration & Production": "oil-and-gas.json",
}

@router.get("/onboarding")
async def onboarding(industry: str):
    file = f"api/service/jsons/{file_map.get(industry, 'software-and-it.json')}"
    with open(file) as f:
        data = json.load(f)
    return JSONResponse({"response": data})



def setup(app):
    app.include_router(router)
