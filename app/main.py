from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging()
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="""
Upload documents, wait for background indexing, then ask grounded questions.

**Choose a chat provider**

* `openai` — hosted OpenAI model with `search_documents` function calling.
* `vllm` — your locally served or Colab-hosted open-source model.
* Omit `provider` — uses `DEFAULT_MODEL_PROVIDER` from `.env`.

Use **GET /api/providers** for the same choices at runtime.
""",
)
app.include_router(router, prefix="/api")
