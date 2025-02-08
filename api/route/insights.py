import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from api.service import chat_handler

router = APIRouter()
prefix = "/insights"

@router.get("/emissions")
async def get_insights_emissions():
    """
    select all rows from database
    calculate scope 1,2,3 emissions for each year
    also get green initiative data from the CSR part
    calculate carbon metric tonnes offset (summation of total trees planted * 22 co2 kilos per tree)
    - return that also
    """
    pass

def setup(app):
    app.include_router(router)