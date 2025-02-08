import json
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from api.service.database import Database, CSRRecord

prefix = "/insights"
router = APIRouter(prefix=prefix)

# Initialize database
db = Database()

class EmissionsResponse(BaseModel):
    total_trees: int
    carbon_offset_kg: float
    carbon_offset_metric_tonnes: float
    csr_records: list

@router.get("/emissions")
async def get_insights_emissions(user_id: str, db_session: Session = Depends(db.get_db)):
    """
    Calculate carbon offset metrics based on tree planting initiatives from CSR records.
    
    Args:
        user_id: The ID of the user whose CSR records to analyze
        db_session: Database session dependency
    
    Returns:
        JSON response with total trees planted and carbon offset calculations
    """
    # Get all CSR records for the user
    records = db_session.query(CSRRecord).filter(CSRRecord.user_id == user_id).all()
    
    print(f"\nFetched {len(records)} CSR records for user_id: {user_id}")
    
    # Debug print each record's metrics
    for idx, record in enumerate(records, 1):
        print(f"\nRecord {idx}:")
        print(f"ID: {record.id}")
        print(f"Name: {record.name}")
        print("Metrics:", json.dumps(record.metrics, indent=2) if record.metrics else "None")
    
    total_trees = 0
    relevant_records = []

    # Process each record
    for record in records:
        if record.metrics and isinstance(record.metrics, list):
            # Look for metrics where unit is "trees"
            for metric in record.metrics:
                if (isinstance(metric, list) and 
                    len(metric) >= 2 and 
                    isinstance(metric[1], str) and 
                    metric[1].lower() == "trees"):
                    try:
                        tree_count = int(metric[0])
                        total_trees += tree_count
                        relevant_records.append({
                            "id": record.id,
                            "name": record.name,
                            "description": record.description,
                            "tree_count": tree_count,
                            "date": record.start_date
                        })
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert tree count '{metric[0]}' to integer")
                        continue

    # Calculate carbon offset
    # 22 kg CO2 per tree
    carbon_offset_kg = total_trees * 22
    carbon_offset_metric_tonnes = carbon_offset_kg / 1000

    response = EmissionsResponse(
        total_trees=total_trees,
        carbon_offset_kg=carbon_offset_kg,
        carbon_offset_metric_tonnes=carbon_offset_metric_tonnes,
    )

    return JSONResponse({
        "status": "success",
        "data": response.dict()
    })

def setup(app):
    app.include_router(router, prefix=prefix)
