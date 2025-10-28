"""Common models used across ingest pipelines."""

from pydantic import BaseModel


class IngestRecord(BaseModel):
    source: str
    path: str
    checksum: str
    status: str = "pending"
