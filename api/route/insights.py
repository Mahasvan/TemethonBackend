import json
from typing import Optional
import csv
from io import StringIO
from fastapi.responses import StreamingResponse

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse
from sqlalchemy import func, desc

from api.service.database import Database, CSRRecord, EmployeeReview, EmployeeEvaluation, User

prefix = "/insights"
router = APIRouter(prefix=prefix)

# Initialize database
db = Database()

class EmissionsResponse(BaseModel):
    total_trees: int
    carbon_offset_kg: float
    carbon_offset_metric_tonnes: float

class DiversityMetricsResponse(BaseModel):
    gender_distribution: dict
    religion_distribution: dict
    disability_percentage: float

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

@router.get("/diversity-metrics")
async def get_diversity_metrics(db_session: Session = Depends(db.get_db)):
    """
    Calculate diversity metrics from employee reviews.
    
    Returns:
        JSON response with distribution of gender, religion and disability status
    """
    # Get total count
    total_count = db_session.query(func.count(EmployeeReview.id)).scalar()
    
    if total_count == 0:
        return JSONResponse({
            "status": "success",
            "data": {
                "gender_distribution": {},
                "religion_distribution": {},
                "disability_percentage": 0.0
            }
        })

    # Get gender distribution
    gender_counts = db_session.query(
        EmployeeReview.gender,
        func.count(EmployeeReview.id)
    ).group_by(EmployeeReview.gender).all()
    
    gender_distribution = {
        gender if gender else "Not Specified": (count/total_count) * 100
        for gender, count in gender_counts
    }

    # Get religion distribution
    religion_counts = db_session.query(
        EmployeeReview.religion,
        func.count(EmployeeReview.id)
    ).group_by(EmployeeReview.religion).all()
    
    religion_distribution = {
        religion if religion else "Not Specified": (count/total_count) * 100
        for religion, count in religion_counts
    }

    # Get disability percentage
    disability_count = db_session.query(func.count(EmployeeReview.id))\
        .filter(EmployeeReview.disability_status == True).scalar()
    disability_percentage = (disability_count/total_count) * 100

    response = DiversityMetricsResponse(
        gender_distribution=gender_distribution,
        religion_distribution=religion_distribution,
        disability_percentage=disability_percentage
    )

    return JSONResponse({
        "status": "success",
        "data": response.dict()
    })

@router.get("/report")
async def get_user_report(user_id: str, db_session: Session = Depends(db.get_db)):
    # Get CSR records for the user
    csr_records = db_session.query(CSRRecord).filter(CSRRecord.user_id == user_id).all()
    
    # Get employee evaluation with highest processed count
    top_evaluation = db_session.query(EmployeeEvaluation).order_by(desc(EmployeeEvaluation.processed_count)).first()
    
    # Get user's onboarding data
    user = db_session.query(User).filter(User.id == user_id).first()
    
    # Create CSV file in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Data Type', 'Field', 'Value'])
    
    # Write CSR Records
    for record in csr_records:
        writer.writerow(['CSR Record', 'ID', record.id])
        writer.writerow(['CSR Record', 'Name', record.name])
        writer.writerow(['CSR Record', 'Description', record.description])
        writer.writerow(['CSR Record', 'Start Date', record.start_date])
        writer.writerow(['CSR Record', 'End Date', record.end_date])
        writer.writerow(['CSR Record', 'Track', record.track])
        writer.writerow(['CSR Record', 'Complete', record.complete])
        
    # Write top evaluation data
    if top_evaluation:
        writer.writerow(['Top Evaluation', 'ID', top_evaluation.id])
        writer.writerow(['Top Evaluation', 'Start DateTime', top_evaluation.start_datetime])
        writer.writerow(['Top Evaluation', 'End DateTime', top_evaluation.end_datetime])
        writer.writerow(['Top Evaluation', 'Total Employees', top_evaluation.total_employees])
        writer.writerow(['Top Evaluation', 'Processed Count', top_evaluation.processed_count])
        writer.writerow(['Top Evaluation', 'Status', top_evaluation.status])
    
    # Write user onboarding data
    if user and user.onboarding_data:
        for key, value in user.onboarding_data.items():
            writer.writerow(['User Onboarding', key, value])
    
    # Prepare the output
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            'Content-Disposition': f'attachment; filename=user_{user_id}_report.csv'
        }
    )

def setup(app):
    app.include_router(router, prefix=prefix)
