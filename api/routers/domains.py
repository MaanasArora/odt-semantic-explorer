from typing import Annotated, List, Optional
from io import BytesIO
import numpy as np
import pandas as pd
from sqlalchemy import select, update, delete, exists
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from settings import Settings, get_settings
from database import get_db
from models import Dataset, Column, Domain


router = APIRouter(prefix="/domains")


class DatasetView(BaseModel):
    id: int
    name: str


class ColumnView(BaseModel):
    dataset: DatasetView
    name: str


class DomainView(BaseModel):
    id: int
    columns: List[ColumnView]


class DomainWithExamples(DomainView):
    examples: List


def get_domain_examples(db: Session, domain_id: int):
    query = select(Column).where(Column.domain_id == domain_id)
    columns = db.execute(query).scalars().all()
    examples = []
    for column in columns:
        dataset_name = column.dataset.name
        df = pd.read_sql_table(dataset_name, db.connection())
        if column.name not in df.columns:
            continue
        values = df[column.name].values
        values = list(set(values))
        samples = np.random.choice(values, 100)
        samples = filter(None, samples)
        examples.extend(samples)
    if len(examples) > 100:
        examples = np.random.choice(examples, 100)
    examples = np.array(examples).astype(str)
    examples = list(set(examples))
    return examples


@router.get("/", response_model=List[DomainView])
def get_domains(db: Annotated[Session, Depends(get_db)]):
    query = select(Domain)
    domains = db.execute(query).scalars().all()
    return domains


@router.get("/{domain_id}", response_model=DomainWithExamples)
def get_domain(db: Annotated[Session, Depends(get_db)], domain_id: int):
    query = select(Domain).where(Domain.id == domain_id)
    domain = db.execute(query).scalar()
    if domain is None:
        raise HTTPException(status_code=404, detail="Domain not found")
    examples = get_domain_examples(db, domain_id)
    return DomainWithExamples.model_validate(
        {"id": domain.id, "examples": examples, "columns": domain.columns},
        from_attributes=True,
    )
