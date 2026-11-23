import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

# Ensure project root (AGROGUARD_AI_ML) is on sys.path so `src.*` imports cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.engine import load_artifacts, run_agroguard_pipeline

app = FastAPI(
    title="AGROGUARD AI Decision Engine",
    description="Calibrated crop intelligence API with Decision Autopsy, Model Arena, Compatibility Frontier, and Decision Passport.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    N: float = Field(..., description="Nitrogen ratio (kg/ha)", example=90.0)
    P: float = Field(..., description="Phosphorus ratio (kg/ha)", example=42.0)
    K: float = Field(..., description="Potassium ratio (kg/ha)", example=43.0)
    temperature: float = Field(..., description="Ambient temperature (°C)", example=20.8)
    humidity: float = Field(..., description="Relative humidity (%)", example=82.0)
    ph: float = Field(..., description="Soil pH level (0–14)", example=6.5)
    rainfall: float = Field(..., description="Rainfall (mm)", example=202.9)
    season: Optional[str] = Field(None, description="Growing season (Kharif, Rabi, Zaid)", example="Kharif")
    region: Optional[str] = Field(None, description="Indian state or agro-climatic zone", example="Telangana")


@app.get("/")
def serve_frontend():
    index_file = ROOT_DIR / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AGROGUARD AI API running. Visit /docs for Swagger UI."}


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "healthy", "model_version": "AGROGUARD-ML-v2.0"}


@app.get("/api/v1/metadata")
def get_model_metadata() -> Dict[str, Any]:
    _, _, _, meta = load_artifacts()
    return {
        "model_version": meta["model_version"],
        "selected_model": meta["selected_model"],
        "classes": meta["classes"],
        "training_min": meta["training_min"],
        "training_max": meta["training_max"],
        "arena_leaderboard": meta["arena_leaderboard"],
        "limitations": meta["limitations"],
    }


@app.post("/api/v1/analyze")
def analyze_soil_and_climate(payload: AnalyzeRequest) -> Dict[str, Any]:
    features = payload.model_dump(exclude={"season", "region"})
    result = run_agroguard_pipeline(features, season=payload.season, region=payload.region)
    if result.get("status") == "INVALID_INPUT":
        raise HTTPException(
            status_code=422,
            detail={"status": "INVALID_INPUT", "errors": result.get("errors", [])},
        )
    return result


@app.post("/predict")
def predict_endpoint(payload: AnalyzeRequest) -> Dict[str, Any]:
    """Alias route matching standard FastAPI /predict Postman testing."""
    return analyze_soil_and_climate(payload)


@app.post("/api/v1/passport/download")
def download_decision_passport(payload: AnalyzeRequest) -> Response:
    features = payload.model_dump(exclude={"season", "region"})
    result = run_agroguard_pipeline(features, season=payload.season, region=payload.region)
    if result.get("status") == "INVALID_INPUT":
        raise HTTPException(status_code=422, detail=result)

    passport = result["decision_passport"]
    filename = f"{passport['passport_id']}.json"
    return Response(
        content=json.dumps(passport, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )