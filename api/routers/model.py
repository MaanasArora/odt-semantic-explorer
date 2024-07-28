from typing import Annotated, List
from io import BytesIO
import pandas as pd
from sqlalchemy import select, update, delete, exists, null
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException
import httpx

from settings import Settings, get_settings
from database import get_db
from models import Dataset, Column, Domain
from algorithm.model import fit_model


router = APIRouter(prefix="/model")


def get_column_data(db: Session):
    query = select(Dataset)
    raw_datasets = db.execute(query).scalars().all()
    column_ids = []
    column_data = []
    for raw_dataset in raw_datasets:
        table_name = raw_dataset.name
        df = pd.read_sql_table(table_name, db.connection())
        df = df.dropna(axis=1, how="all")
        for col in df.columns:
            query = select(Column).where(Column.name == col)
            column = db.execute(query).scalar()
            if column is None:
                column = Column(name=col, dataset_id=raw_dataset.id)
                db.add(column)
                db.commit()
            column_ids.append(column.id)
            column_data.append(df[col])
    return column_ids, column_data


def create_domains(db: Session, column_ids, clusters):
    query = delete(Domain)
    db.execute(query)

    query = update(Column).values(domain_id=null())
    db.execute(query)
    db.commit()

    cluster_counts = pd.Series(clusters).value_counts()
    all_domains = cluster_counts[cluster_counts > 1].index.unique()
    
    for domain in all_domains:
        domain = Domain(id=domain)
        db.add(domain)

    for column_id, cluster in zip(column_ids, clusters):
        if cluster not in all_domains:
            continue
        query = update(Column).where(Column.id == column_id).values(domain_id=cluster)
        db.execute(query)

    db.commit()
    return len(all_domains)


@router.post("/fit")
def fit_clusters(db: Annotated[Session, Depends(get_db)]):
    column_ids, column_data = get_column_data(db)
    clusters = fit_model(column_data)
    num_domains = create_domains(db, column_ids, clusters)
    return {"domains": {"count": num_domains}}
