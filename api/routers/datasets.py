from typing import Annotated, List
from io import BytesIO
import pandas as pd
from sqlalchemy import select, exists
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException
import httpx
from tqdm import tqdm

from settings import Settings, get_settings
from database import get_db
from models import Dataset, Column


router = APIRouter(prefix="/datasets")


def get_dataset_name_from_filename(filename: str):
    if filename[0].isnumeric():
        filename = f"_{filename}"
    filename = filename.strip(".csv").lower()
    filename = filename.replace("-", "_").replace(".", "_").replace(" ", "_")
    return filename


def get_dataset_exists(db: Session, filename: str):
    dataset_name = get_dataset_name_from_filename(filename)
    query = select(exists().where(Dataset.name == dataset_name))
    dataset = db.execute(query).scalar()
    return dataset


def create_raw_dataset_from_csv(content, filename: str, db: Session, nrows=20000):
    content_io = BytesIO(content)
    df = pd.read_csv(content_io)

    df.columns = df.columns.str.lower()
    df.columns = [col if len(col) <= 50 else col[:50] for col in df.columns]
    no_duplicate_cols = set()
    for col in df.columns:
        if col not in no_duplicate_cols:
            no_duplicate_cols.add(col)
        else:
            i = 1
            while f"{col}_{i}" in no_duplicate_cols:
                i += 1
            no_duplicate_cols.add(f"{col}_{i}")
    df.columns = no_duplicate_cols

    filename = get_dataset_name_from_filename(filename)

    if nrows:
        if df.shape[0] > nrows:
            df = df.sample(nrows)
    df.to_sql(filename, db.connection(), if_exists="replace", index=False)

    dataset = Dataset(name=filename, size=len(content), num_rows=len(df))
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    column_names = df.columns
    for column_name in column_names:
        column = Column(name=column_name, dataset_id=dataset.id)
        db.add(column)

    db.commit()

    return filename


def get_open_dataset_ids(limit: int = None, offset: int = 0):
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = f"{base_url}/api/3/action/package_list"

    response = httpx.get(url, params={"limit": limit, "offset": offset})
    response.raise_for_status()

    data = response.json()
    return data["result"]


def fetch_open_dataset(id: str, settings: Settings, db: Session):
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = f"{base_url}/api/3/action/package_show?id={id}"

    response = httpx.get(url)
    response.raise_for_status()

    data = response.json()
    dataset = data["result"]
    resources = dataset["resources"]

    filenames = []
    for resource in resources:
        if resource["format"].lower() != "csv":
            continue

        url = resource["url"]
        response = httpx.get(url)
        if response.status_code != 200:
            print(f"Error getting details for dataset: {url}")
            continue

        filename = resource["name"]
        if len(filename) > 50:
            filename = filename[:50]

        if get_dataset_exists(db, filename):
            print(f"Dataset already exists: {filename}")
            continue

        content = response.content

        try:
            filename_raw = create_raw_dataset_from_csv(content, filename, db)
        except Exception as e:
            print(f"Error creating dataset: {filename}")
            print(e)
            continue

        filenames.append(filename_raw)
        break
    return filenames


@router.get("/")
def read_datasets(
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Session, Depends(get_db)],
):
    query = select(Dataset)
    datasets = db.execute(query).scalars().all()
    return datasets


@router.get("/{name}")
def read_dataset(
    name: str,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Session, Depends(get_db)],
):
    query = select(Dataset).where(Dataset.name == name)
    dataset = db.execute(query).scalars().first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    data = pd.read_sql_table(name, db.connection())
    return {
        "name": dataset.name,
        "size": dataset.size,
        "num_rows": dataset.num_rows,
        "data": data,
    }


@router.post("/open/crawl")
def crawl_datasets(
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Session, Depends(get_db)],
    download: bool = False,
    limit: int = 10,
    offset: int = 0,
):
    ids = get_open_dataset_ids(limit, offset)

    if not download:
        return {"found": {"count": len(ids), "ids": ids}}

    filenames = []
    for id in tqdm(ids):
        filenames += fetch_open_dataset(id, settings, db)

    return {
        "found": {"count": len(ids), "ids": ids},
        "downloaded": {"count": len(filenames), "filenames": filenames},
    }


@router.post("/open/{id}")
def download_dataset(
    id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Session, Depends(get_db)],
):
    filenames = fetch_open_dataset(id, settings, db)
    return {"count": len(filenames), "filenames": filenames}
