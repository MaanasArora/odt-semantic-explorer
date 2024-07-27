from typing import Annotated, List
from io import BytesIO
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, HTTPException
import httpx

from routers.datasets import router as datasets_router
from routers.model import router as model_router
from routers.domains import router as domains_router


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


app.include_router(datasets_router, tags=["datasets"])
app.include_router(model_router, tags=["model"])
app.include_router(domains_router, tags=["domains"])
