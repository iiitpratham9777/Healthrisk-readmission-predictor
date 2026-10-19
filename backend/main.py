"""FastAPI backend.  Run:  uvicorn backend.main:app --reload"""
import json, sys
from pathlib import Path
import joblib, pandas as pd, xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from common import DISPOSITIONS, DX_CATEGORIES, engineer  # noqa: E402

ART = Path(__file__).parent / "artifacts"
if not (ART / "model.joblib").exists():
    raise SystemExit("Model not found. Run `python ml/train.py` first.")
bundle = joblib.load(ART / "model.joblib")
model, THR, MED = bundle["model"], bundle["threshold"], bundle["comorb_median"]
METRICS = json.load(open(ART / "metrics.json"))
NAMES = bundle["feature_names"]

app = FastAPI(title="Hospital Readmission Risk API", version="1.0")


class Patient(BaseModel):
    age: int = Field(65, ge=18, le=110)
    gender: str = Field("Female", pattern="^(Male|Female)$")
    length_of_stay: int = Field(4, ge=1, le=60)
    emergency_admission: int = Field(0, ge=0, le=1)
    num_diagnoses: int = Field(5, ge=1, le=25)
    num_procedures: int = Field(1, ge=0, le=15)
    num_medications: int = Field(8, ge=0, le=50)
    prior_admissions_1yr: int = Field(0, ge=0, le=20)
    prior_ed_visits_1yr: int = Field(1, ge=0, le=25)
    comorbidity_score: int = Field(2, ge=0, le=15)
    abnormal_creatinine: int = Field(0, ge=0, le=1)
    abnormal_hba1c: int = Field(0, ge=0, le=1)
    abnormal_hemoglobin: int = Field(0, ge=0, le=1)
    discharge_disposition: str = "Home"
    primary_dx_category: str = "Other"


def tier(p):
    return "High" if p >= THR else ("Moderate" if p >= THR * 0.75 else "Low")


@app.get("/api/health")
def health():
    return {"status": "ok", "threshold": THR}


@app.get("/api/metadata")
def metadata():
    return {"dispositions": DISPOSITIONS, "dx_categories": DX_CATEGORIES, "threshold": THR}


@app.get("/api/metrics")
def metrics():
    return METRICS


@app.post("/api/predict")
def predict(p: Patient):
    if p.discharge_disposition not in DISPOSITIONS or p.primary_dx_category not in DX_CATEGORIES:
        raise HTTPException(422, "Unknown discharge_disposition or primary_dx_category")
    X = engineer(pd.DataFrame([p.model_dump()]), MED)
    prob = float(model.predict_proba(X)[0, 1])
    # Per-patient explanation: XGBoost SHAP-style contributions (log-odds) on transformed features
    Xt = model.named_steps["pre"].transform(X)
    contrib = model.named_steps["clf"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0][:-1]
    top = sorted(zip(NAMES, contrib), key=lambda t: -abs(t[1]))[:6]
    return {"probability": prob, "risk_tier": tier(prob), "threshold": THR, "flagged": prob >= THR,
            "drivers": [{"feature": f, "impact": float(c)} for f, c in top]}


app.mount("/assets", StaticFiles(directory=ROOT / "frontend"), name="assets")


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")
