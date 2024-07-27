from typing import List, Optional
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class RawDataset(Base):
    __tablename__ = "raw_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    size: Mapped[float] = mapped_column()
    num_rows: Mapped[int] = mapped_column()

    def __repr__(self):
        return f"<RawDataset {self.name}>"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    size: Mapped[float] = mapped_column()
    num_rows: Mapped[int] = mapped_column()

    def __repr__(self):
        return f"<Dataset {self.name}>"
