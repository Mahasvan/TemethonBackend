from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.service.governance_analyzer import GovernanceAnalyzer
from typing import Dict
import pprint

router = APIRouter()
governance_analyzer = GovernanceAnalyzer()

class CompanyRequest(BaseModel):
    company: str
    industry: str

@router.post("/analyze-governance")
async def analyze_governance(request: CompanyRequest) -> Dict:
    """
    Analyze governance aspects and government stance on CSR activities for a company.
    Returns detailed analysis of regulatory impact, governance landscape, and ESG governance score.
    """
    try:
        result = governance_analyzer.aggregate_governance_analysis(
            company=request.company,
            industry=request.industry
        )
        pprint.pprint(result)  # For debugging/logging
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def setup(app):
    app.include_router(router, prefix="/governance", tags=["Governance Analysis"]) 