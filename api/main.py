from typing import Annotated, List
from io import BytesIO
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends, HTTPException
import httpx

from routers.datasets import router as datasets_router


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


app.include_router(datasets_router)
