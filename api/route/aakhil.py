from fastapi.routing import APIRouter
from fastapi import Depends

import uuid, json

from api.service import database
from api.service.models import CSREvent, CSRRequest
from api.service.database import CSRRecord, User, Database
from api.service.chat_handler import initiative_classification_chain as classification_chain

from fastapi.exceptions import HTTPException

from sqlalchemy.orm import Session

db = Database()
get_db = db.get_db

prefix = "/aakhil"
router = APIRouter(prefix=prefix)


@router.post("/classify", response_model=CSREvent)
async def classify_csr_event(request: CSRRequest, db: Session = Depends(get_db)):
    try:
        print("\n=== Starting CSR Classification ===")
        print(f"Request data: {request.dict()}")

        # Check for user if user_id is provided
        if request.user_id:
            print(f"\nLooking up user: {request.user_id}")
            user = db.query(User).filter(User.id == request.user_id).first()
            if not user:
                raise HTTPException(
                    status_code=404,
                    detail=f"User not found with ID: {request.user_id}"
                )
            print(f"Found user: {user.id}")

        if request.event_id:
            print(f"\nProcessing existing event: {request.event_id}")
            record = db.query(CSRRecord).filter(CSRRecord.id == request.event_id).first()
            if not record:
                raise HTTPException(
                    status_code=404,
                    detail=f"Event not found with ID: {request.event_id}"
                )

            print(f"Found record: {record.id}")

            if record.questions and request.followup_answers:
                print("\nProcessing follow-up answers")
                qa_history = record.qa_history or []
                for q, a in zip(record.questions, request.followup_answers):
                    qa_history.append([q, a])
                record.qa_history = qa_history
                print(f"Updated QA history: {qa_history}")

            qa_pairs = []
            if record.qa_history:
                for q, a in record.qa_history:
                    qa_pairs.append(f"Question: {q}")
                    qa_pairs.append(f"Answer: {a}")

            description = f"""Original event: {record.description}

Previous questions and answers:
{chr(10).join(qa_pairs)}"""

            record_id = request.event_id
        else:
            print("\nCreating new record")
            record_id = str(uuid.uuid4())
            record = CSRRecord(
                id=record_id,
                user_id=request.user_id
            )
            db.add(record)
            description = request.description
            print(f"Created new record: {record_id}")

        print("\nCalling LLM for classification...")
        print(f"Input description: {description}")

        response = await classification_chain.ainvoke(
            input={"description": description}
        )

        print(f"\nLLM Response received: {response}")

        try:
            print("\nParsing LLM response...")
            response_text = response.content.strip()
            if '```' in response_text:
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]

            print(f"Cleaned response text: {response_text}")

            result_dict = json.loads(response_text.strip())
            result_dict['id'] = record_id

            # Add default values for incomplete responses
            if not result_dict.get('complete', True):
                result_dict.update({
                    'name': None,
                    'description': description,
                    'start_date': None,
                    'end_date': None,
                    'attendees': None,
                    'track': None,
                    'metrics': [],
                })

            print(f"\nParsed result with defaults: {result_dict}")

            result = CSREvent(**result_dict)
            print("\nCreated CSREvent object")

            print("\nUpdating record with results...")
            record.name = result.name
            record.description = description
            record.start_date = result.start_date
            record.end_date = result.end_date
            record.attendees = result.attendees
            record.track = result.track
            record.metrics = result.metrics
            record.complete = result.complete
            record.questions = result.questions

            try:
                print("\nCommitting to database...")
                db.commit()
                print(f"Successfully committed record: {record_id}")
            except Exception as e:
                db.rollback()
                print(f"Database commit failed: {str(e)}")
                print(f"Full error: {repr(e)}")
                raise

            return result

        except Exception as e:
            db.rollback()
            print(f"\nError processing LLM response: {str(e)}")
            print(f"Full error: {repr(e)}")
            print(f"Response text was: {response['text']}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM response: {str(e)}\nResponse: {response['text']}"
            )

    except Exception as e:
        if 'db' in locals():
            db.rollback()
        print(f"\nUnhandled error: {str(e)}")
        print(f"Full error: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/status/{event_id}", response_model=dict)
async def get_event_status(event_id: str, db: Session = Depends(get_db)):
    try:
        record = db.query(CSRRecord).filter(CSRRecord.id == event_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Event not found")

        return {
            "id": record.id,
            "complete": record.complete,
            "initial_description": record.description,
            "current_questions": record.questions if not record.complete else None,
            "qa_history": record.qa_history or [],
            "current_data": {
                "name": record.name,
                "description": record.description,
                "start_date": record.start_date,
                "end_date": record.end_date,
                "attendees": record.attendees,
                "track": record.track,
                "metrics": record.metrics
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching status: {str(e)}"
        )
