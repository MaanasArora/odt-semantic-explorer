from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:postgres@localhost:5432/toronto"
    allowed_origins: List[str] = ["http://localhost:5173"]

    class Config:
        env_file = ".env"


settings = Settings()


def get_settings():
    return Settings()
