from typing import List, Optional
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    size: Mapped[float] = mapped_column()
    num_rows: Mapped[int] = mapped_column()

    columns = relationship("Column", backref="dataset")

    def __repr__(self):
        return f"<Dataset {self.name}>"


class Column(Base):
    __tablename__ = "columns"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    domain_id: Mapped[Optional[int]] = mapped_column(ForeignKey("domains.id"))
    name: Mapped[str] = mapped_column()

    def __repr__(self):
        return f"<Column {self.name}>"


class Domain(Base):
    __tablename__ = "domains"

    id: Mapped[int] = mapped_column(primary_key=True)

    columns = relationship("Column", backref="domain")

    def __repr__(self):
        return f"<Domain {self.id}>"
