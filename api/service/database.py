from sqlalchemy import create_engine, Column, String, Boolean, JSON, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime
import uuid
from sqlalchemy.dialects.postgresql import JSONB

DATABASE_URL = "postgresql://postgres:1ji2HYYOkYAmQIEP6s8nvEbTEYHJSe3CaS78BZVnoZOIWo70CZ7t66q6bNMcR6ZZ@5.78.129.71:14028/postgres"

Base = declarative_base()

class CSRRecord(Base):
    __tablename__ = "csr_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    start_date = Column(String, nullable=True)
    end_date = Column(String, nullable=True)
    attendees = Column(String, nullable=True)
    track = Column(String, nullable=True)
    metrics = Column(JSON, nullable=True)
    complete = Column(Boolean, default=False)
    questions = Column(JSON, nullable=True)
    qa_history = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EmployeeReview(Base):
    __tablename__ = "employee_reviews"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=True)
    place = Column(String, nullable=True)
    job_type = Column(String, nullable=True)
    department = Column(String, nullable=True)
    date = Column(String, nullable=True)
    overall_rating = Column(String, nullable=True)
    work_life_balance = Column(String, nullable=True)
    skill_development = Column(String, nullable=True)
    salary_and_benefits = Column(String, nullable=True)
    job_security = Column(String, nullable=True)
    career_growth = Column(String, nullable=True)
    work_satisfaction = Column(String, nullable=True)
    likes = Column(String, nullable=True)
    dislikes = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    disability_status = Column(Boolean, default=False)
    religion = Column(String, nullable=True)
    gender = Column(String, nullable=True)

class EmployeeEvaluation(Base):
    __tablename__ = "employee_evaluations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    start_datetime = Column(DateTime, default=datetime.utcnow)
    end_datetime = Column(DateTime, nullable=True)
    total_employees = Column(Integer, default=0)
    processed_count = Column(Integer, default=0)
    evaluated_records = Column(JSON, default=list)
    problem_categories = Column(JSON, default=dict)
    positive_categories = Column(JSON, default=dict)
    status = Column(String, default="in_progress")
    created_at = Column(DateTime, default=datetime.utcnow)

class User(Base):
    __tablename__ = "User"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    email = Column(String, nullable=True, unique=True)
    emailVerified = Column(DateTime, nullable=True)
    password = Column(String, nullable=True)
    image = Column(String, nullable=True)
    companyName = Column(String, nullable=True)
    mission = Column(String, nullable=True)
    vision = Column(String, nullable=True)
    sector = Column(String, nullable=True)
    onboarding_data = Column(JSONB, nullable=True)


class Database:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            Database._instance = object.__new__(cls)
            Database._instance.__init__()
        return cls._instance

    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        # Only create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def create(self, model_class, **fields):
        """
        Create a new record
        Example: db.create(User, name="John", email="john@example.com")
        """
        db = self.SessionLocal()
        try:
            instance = model_class(**fields)
            db.add(instance)
            db.commit()
            db.refresh(instance)
            return instance
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def read(self, model_class, filters=None, single=False):
        """
        Read records with optional filters
        Example: db.read(User, {"email": "john@example.com"}, single=True)
        """
        db = self.SessionLocal()
        try:
            query = db.query(model_class)
            if filters:
                for key, value in filters.items():
                    query = query.filter(getattr(model_class, key) == value)

            if single:
                return query.first()
            return query.all()
        finally:
            db.close()

    def update(self, model_class, filters, **updates):
        """
        Update records matching filters with new values
        Example: db.update(User, {"id": "123"}, name="New Name")
        """
        db = self.SessionLocal()
        try:
            query = db.query(model_class)
            for key, value in filters.items():
                query = query.filter(getattr(model_class, key) == value)

            instance = query.first()
            if not instance:
                return None

            for key, value in updates.items():
                setattr(instance, key, value)

            db.commit()
            db.refresh(instance)
            return instance
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def delete(self, model_class, filters):
        """
        Delete records matching filters
        Example: db.delete(User, {"id": "123"})
        """
        db = self.SessionLocal()
        try:
            query = db.query(model_class)
            for key, value in filters.items():
                query = query.filter(getattr(model_class, key) == value)

            instance = query.first()
            if instance:
                db.delete(instance)
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def bulk_create(self, model_class, records):
        """
        Create multiple records at once
        Example: db.bulk_create(User, [{"name": "John"}, {"name": "Jane"}])
        """
        db = self.SessionLocal()
        try:
            instances = [model_class(**record) for record in records]
            db.bulk_save_objects(instances)
            db.commit()
            return instances
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def bulk_update(self, model_class, filters, updates):
        """
        Update multiple records at once
        Example: db.bulk_update(User, {"role": "user"}, {"active": True})
        """
        db = self.SessionLocal()
        try:
            query = db.query(model_class)
            for key, value in filters.items():
                query = query.filter(getattr(model_class, key) == value)
            query.update(updates)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()