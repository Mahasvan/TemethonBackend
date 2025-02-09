from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.service.knowledge_aggregator import KnowledgeAggregator
from typing import Dict
import pprint
router = APIRouter()
knowledge_aggregator = KnowledgeAggregator()

class CompanyRequest(BaseModel):
    company: str
    industry: str

@router.post("/aggregate-knowledge")
async def aggregate_company_knowledge(request: CompanyRequest) -> Dict:
    """
    Aggregate knowledge about a company's CSR activities and its competitors.
    """
    try:
        result = knowledge_aggregator.aggregate_knowledge(
            company=request.company,
            industry=request.industry
        )
        pprint.pprint(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def setup(app):
    app.include_router(router, prefix="/knowledge", tags=["Knowledge Aggregation"]) 