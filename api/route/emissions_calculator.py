from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.service.emissions_calculator import EmissionsCalculator
from typing import Dict
import pprint

router = APIRouter()
emissions_calculator = EmissionsCalculator()

class EmissionsRequest(BaseModel):
    employees_count: int
    wfh_employees_count: int
    electricity_consumption_kwh: float  # Total electricity consumption in kWh
    td_losses: float  # transmission and distribution losses in metric tonnes
    waste_emissions: float  # waste emissions in metric tonnes
    location: str

@router.post("/calculate-emissions")
async def calculate_emissions(request: EmissionsRequest) -> Dict:
    """
    Calculate CO2 emissions in metric tonnes based on company data.
    Takes into account:
    - Electricity consumption (distributed by country's energy mix)
    - Employee commute
    - Work from home emissions
    - Transmission and distribution losses
    - Waste emissions
    """
    try:
        # Prepare data for calculation
        data = {
            "employees_count": request.employees_count,
            "wfh_employees_count": request.wfh_employees_count,
            "electricity_consumption_kwh": request.electricity_consumption_kwh,
            "td_losses": request.td_losses,
            "waste_emissions": request.waste_emissions,
            "location": request.location
        }
        
        # Calculate emissions
        result = emissions_calculator.calculate_emissions(data)
        pprint.pprint(result)  # For debugging/logging
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def setup(app):
    app.include_router(router, prefix="/emissions", tags=["Emissions Calculator"]) 