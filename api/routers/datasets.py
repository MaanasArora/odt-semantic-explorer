from typing import Annotated, List
from io import BytesIO
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException
import httpx

from settings import Settings, get_settings
from database import get_db
from models import Dataset


router = APIRouter(prefix="/datasets")


def get_dataset_name_from_filename(filename: str):
    return (
        f"{filename}_raw".replace("-", "_").replace(".", "_").replace(" ", "_").lower()
    )


def create_raw_dataset_from_csv(content, filename: str, db: Session):
    content_io = BytesIO(content)
    df = pd.read_csv(content_io)

    filename_raw = get_dataset_name_from_filename(filename)
    df.to_sql(filename_raw, db.connection(), if_exists="replace", index=False)

    dataset = RawDataset(name=filename_raw, size=len(content), num_rows=len(df))
    db.add(dataset)
    db.commit()

    return filename_raw


def get_open_dataset_ids():
    base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
    url = f"{base_url}/api/3/action/package_list"

    response = httpx.get(url)
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
        response.raise_for_status()

        filename = resource["name"]
        content = response.content

        try:
            filename_raw = create_raw_dataset_from_csv(content, filename, db)
        except Exception as e:
            print(f"Error creating dataset: {filename}")
            print(e)
            continue

        filenames.append(filename_raw)
    return filenames


@router.get("/")
def read_datasets(
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Session, Depends(get_db)],
):
    datasets = db.query(Dataset).all()
    return datasets


@router.get("/{name}")
def read_dataset(
    name: str,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Session, Depends(get_db)],
):
    dataset = db.query(Dataset).filter(Dataset.name == name).first()
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
):
    ids = get_open_dataset_ids()

    if limit:
        if len(ids) > limit:
            ids = ids[:limit]

    if not download:
        return {"count": len(ids), "ids": ids}

    filenames = []
    for id in ids:
        filenames += fetch_open_dataset(id, settings, db)

    return {"count": len(filenames), "filenames": filenames}


@router.post("/open/{id}")
def download_dataset(
    id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    db: Annotated[Session, Depends(get_db)],
):
    filenames = fetch_open_dataset(id, settings, db)
    return {"count": len(filenames), "filenames": filenames}
