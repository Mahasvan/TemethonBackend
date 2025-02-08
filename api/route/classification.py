import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from api.service import chat_handler
from api.service.database import Database, CSRRecord

prefix = "/classification"
router = APIRouter(prefix=prefix)

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
    """Get a materiality assessment based on company description."""
    try:
        print(f"Input description: {description.description}")  # Debug log
        response = chat_handler.materiality_assessment_chain.invoke(input={
            "description": description.description,
        })
        print(f"LLM Response: {response}")  # Debug log
        return JSONResponse({
            "response": json.loads(response["text"])
        })
    except Exception as e:
        print(f"Error in materiality assessment: {str(e)}")  # Debug log
        raise

@router.get("/report-format/{user_id}")
async def get_report_format(user_id: str):
    # Initialize database
    db = Database()
    
    # Get list of initiatives filtered by user_id
    initiatives = db.read(CSRRecord, filters={"user_id": user_id})
    
    # Prepare data for the report structure chain
    data = {
        "initiatives": [
            {
                "name": initiative.name,
                "description": initiative.description,
                "start_date": initiative.start_date,
                "end_date": initiative.end_date,
                "track": initiative.track,
                "metrics": initiative.metrics
            }
            for initiative in initiatives
        ]
    }
    
    # Feed data to LLM
    response = chat_handler.report_structure_chain.invoke(input={
        "data": data
    })
    print(response["text"])
    return {
        "report_structure": response["text"]
    }

def setup(app):
    app.include_router(router)
