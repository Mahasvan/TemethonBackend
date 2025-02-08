import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from api.service import chat_handler

router = APIRouter()
prefix = "/forms"

class Form(BaseModel):
    data: dict

@router.post("/onboarding-form")
async def onboarding_form(form: Form):
    data = form.data
    """Need to update this into the database"""