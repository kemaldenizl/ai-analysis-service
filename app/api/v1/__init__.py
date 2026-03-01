from fastapi import APIRouter

analysis_router = APIRouter()

@analysis_router.get("/status")
async def analysis_status():
    return {"message": "Analysis engine router is ready."}