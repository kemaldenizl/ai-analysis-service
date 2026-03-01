from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def analysis_status():
    return {"message": "Analysis engine router is ready."}