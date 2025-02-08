import json

from fastapi import APIRouter
from starlette.responses import JSONResponse

from api.service import chat_handler

router = APIRouter()
prefix = "/classification"

industries = ["Biofuels",
              "Biotechnology & Pharmaceuticals",
              "Software & IT Services",
              "Food Retailers & Distributors",
              "Oil & Gas – Exploration & Production"]

@router.get("/industry")
async def classify_industry(description):
    """Identify a company's industry based on its mission statement and description."""
    response = chat_handler.industry_classification_chain.invoke(input={
        "description": description,
    })
    industry = industries[int(response['text'])]
    return {
        "industry": industry,
    }

@router.get("/initiative")
async def classify_csr_initiative(description):
    """extracts information from a CSR Initiative's writeup"""
    response = chat_handler.initiative_classification_chain.invoke(input={
        "description": description,
    })
    return JSONResponse({
        "response": json.loads(response['text'])
    })

def setup(app):
    app.include_router(router)
