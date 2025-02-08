import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from api.service import chat_handler

router = APIRouter()
prefix = "/classification"

class Input(BaseModel):
    description: str

industries = ["Biofuels",
              "Biotechnology & Pharmaceuticals",
              "Software & IT Services",
              "Food Retailers & Distributors",
              "Oil & Gas – Exploration & Production"]

@router.post("/industry")
async def classify_industry(description: Input):
    """Identify a company's industry based on its mission statement and description."""
    response = chat_handler.industry_classification_chain.invoke(input={
        "description": description.description,
    })
    print(response)
    industry = industries[int(response["text"])]
    return {
        "industry": industry,
    }

@router.post("/initiative")
async def classify_csr_initiative(description: Input):
    """extracts information from a CSR Initiative's writeup"""
    response = chat_handler.initiative_classification_chain.invoke(input={
        "description": description.description,
    })
    return JSONResponse({
        "response": json.loads(response["text"])
    })

@router.post("/materiality-assessment")
async def get_materiality_assessment(description: Input):
    # prompt_thing
    response = None # populate here
    """this should be a list of dictionaries - with each element in the list having the following schema
    {
        "name": "Energy",
        "description": "Energy is the main resource requirement for an IT industry bla bla"
    }
    
    same thing for each entry in the list - ordered by importance.
    """

    return JSONResponse({
        "response": response
    })

def setup(app):
    app.include_router(router)
