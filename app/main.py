from fastapi import FastAPI
from app.api.v1 import analysis_router
from app.core.config import settings

app = FastAPI(
    title="AI-Powered Finance Analysis Engine",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None
)

app.include_router(analysis_router, prefix="/api/v1/analysis", tags=["Analysis"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine": "Python FastAPI"}