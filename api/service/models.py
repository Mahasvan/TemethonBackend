from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
from datetime import datetime

class CSREvent(BaseModel):
    id: str = Field(description="Unique identifier for the event")
    name: Optional[str] = Field(description="Name of the CSR event")
    description: Optional[str] = Field(description="Brief description of the event")
    start_date: Optional[str] = Field(description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(description="End date in YYYY-MM-DD format")
    attendees: Optional[int] = Field(description="Number of people attending")
    track: Optional[str] = Field(description="The CSR track this event belongs to")
    metrics: List[List[str]] = Field(description="List of metrics, each containing [quantity, description]")
    complete: bool = Field(description="Whether the event data is complete")
    questions: Optional[List[str]] = Field(description="Questions needed to complete the event data")

class CSRRequest(BaseModel):
    description: Optional[str] = None
    event_id: Optional[str] = None
    followup_answers: Optional[List[str]] = None