from typing import Literal

from pydantic import BaseModel


class DocumentIngestionAccepted(BaseModel):
    document_id: str
    filename: str
    job_id: str
    status: Literal["queued"] = "queued"
