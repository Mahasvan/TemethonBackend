from fastapi import APIRouter

from api.service import chat_handler

router = APIRouter()
prefix = "/classification"

industries = ["Biofuels",
              "Biotechnology & Pharmaceuticals",
              "Software & IT Services",
              "Food Retailers & Distributors",
              "Oil & Gas – Exploration & Production"]

@router.get("/classify")
async def classify(description):
    """Identify a company's industry based on its mission statement and description."""
    response = chat_handler.industry_classification_chain.invoke(input={
        "description": description,
    })
    industry = industries[int(response['text'])]
    return {
        "industry": industry,
    }

def setup(app):
    app.include_router(router)
