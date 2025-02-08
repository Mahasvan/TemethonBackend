import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from api.service.database import Database, User

prefix = "/forms"
router = APIRouter(prefix=prefix)

# Initialize database
db = Database()

def to_snake_case(text):
    """Convert a string to snake_case"""
    import re
    text = re.sub(r'[\-\s]+', '_', text.lower())
    return re.sub(r'[^\w_]', '', text)

def create_field_mapping(form_structure):
    """Create a mapping between snake_case keys and original keys"""
    mapping = {}
    for category, fields in form_structure.items():
        snake_category = to_snake_case(category)
        mapping[snake_category] = {
            'original': category,
            'fields': {to_snake_case(field): field for field in fields.keys()}
        }
    return mapping

class Form(BaseModel):
    data: dict
    user_id: str
    industry: str

@router.get("/form-structure/{industry}")
async def get_form_structure(industry: str):
    """Get the form structure with snake_case keys"""
    try:
        file_map = {
            "Biofuels": "biofuels.json",
            "Biotechnology & Pharmaceuticals": "biotech.json",
            "Software & IT Services": "software-and-it.json",
            "Food Retailers & Distributors": "food-retail.json",
            "Oil & Gas – Exploration & Production": "oil-and-gas.json",
        }
        
        file_name = file_map.get(industry)
        if not file_name:
            raise HTTPException(status_code=400, detail="Invalid industry")

        with open(f"api/service/jsons/{file_name}") as f:
            form_structure = json.load(f)

        # Convert to snake_case structure
        snake_structure = {}
        for category, fields in form_structure.items():
            snake_category = to_snake_case(category)
            snake_structure[snake_category] = {}
            for field_name, field_spec in fields.items():
                snake_field = to_snake_case(field_name)
                snake_structure[snake_category][snake_field] = {
                    "original_name": field_name,
                    "unit": field_spec["Unit of Measure"],
                    "category": field_spec["Category"],
                    "code": field_spec["Code"]
                }

        return JSONResponse({"structure": snake_structure})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/onboarding-form")
async def onboarding_form(form: Form, db_session: Session = Depends(db.get_db)):
    """Save onboarding form data to user's profile"""
    try:
        # Get industry-specific form structure
        file_map = {
            "Biofuels": "biofuels.json",
            "Biotechnology & Pharmaceuticals": "biotech.json",
            "Software & IT Services": "software-and-it.json",
            "Food Retailers & Distributors": "food-retail.json",
            "Oil & Gas – Exploration & Production": "oil-and-gas.json",
        }
        
        # Verify user exists
        user = db_session.query(User).filter(User.id == form.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Load the appropriate industry form structure
        file_name = file_map.get(form.industry)
        if not file_name:
            raise HTTPException(status_code=400, detail="Invalid industry")

        with open(f"api/service/jsons/{file_name}") as f:
            form_structure = json.load(f)

        # Create field mapping
        field_mapping = create_field_mapping(form_structure)

        # Validate and process form data against structure
        processed_data = {}
        for snake_category, category_data in form.data.items():
            if snake_category not in field_mapping:
                continue

            original_category = field_mapping[snake_category]['original']
            processed_data[original_category] = {}
            
            for snake_field, value in category_data.items():
                if snake_field not in field_mapping[snake_category]['fields']:
                    continue

                original_field = field_mapping[snake_category]['fields'][snake_field]
                field_spec = form_structure[original_category][original_field]
                
                processed_data[original_category][original_field] = {
                    "value": value,
                    "unit": field_spec["Unit of Measure"],
                    "category": field_spec["Category"],
                    "code": field_spec["Code"]
                }

        # Update user's onboarding data
        user.onboarding_data = processed_data
        db_session.commit()

        return JSONResponse({
            "status": "success",
            "message": "Onboarding data saved successfully"
        })

    except Exception as e:
        db_session.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def setup(app):
    app.include_router(router)