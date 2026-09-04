from typing import Literal

from pydantic import BaseModel, Field


class DemoBatchRequest(BaseModel):
    """Temporary payload that demonstrates the job mechanism before file ingestion."""

    documents: list[str] = Field(min_length=1, max_length=100)


class JobCreated(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: str
    result: dict | None = None

