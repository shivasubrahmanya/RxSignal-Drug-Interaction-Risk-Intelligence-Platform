import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="FAERS AI Backend")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(base_dir, "models")
processed_dir = os.path.join(base_dir, "processed")

ml_model = None
encoders = None
aggregated_df = None

@app.on_event("startup")
async def startup_event():
    global ml_model, encoders, aggregated_df
    print("Loading ML models and historical data...")
    
    try:
        ml_model = joblib.load(os.path.join(models_dir, "xgboost_risk_model.pkl"))
        encoders = joblib.load(os.path.join(models_dir, "encoders.pkl"))
        print("ML Models loaded.")
    except Exception as e:
        print(f"Warning: ML Models not found or failed to load. {e}")
        
    try:
        # Load the aggregated stats (43M rows). For memory constraints, we'll use duckdb in the query endpoint instead of pandas if it's too big.
        # Actually, let's load it dynamically using duckdb per request for speed and memory efficiency.
        print("Backend ready.")
    except Exception as e:
        print(f"Warning: Data not loaded. {e}")

class PredictionRequest(BaseModel):
    drug_a: str
    drug_b: str

@app.get("/api/drugs")
async def get_drugs():
    if not encoders:
        raise HTTPException(status_code=500, detail="Models not loaded")
    # Return top 1000 drugs for the dropdowns (to avoid massive payload)
    all_drugs = list(encoders['freq'].keys())
    return {"drugs": sorted(all_drugs[:1000])}

@app.post("/api/predict")
async def predict_risk(request: PredictionRequest):
    if not ml_model or not encoders:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    d_a, d_b = sorted([request.drug_a.upper().strip(), request.drug_b.upper().strip()])
    
    # 1. Get ML Prediction
    features = pd.DataFrame([{
        'drug_a_freq': encoders['freq'].get(d_a, 0),
        'drug_b_freq': encoders['freq'].get(d_b, 0),
        'drug_a_mean_risk': encoders['mean_risk'].get(d_a, 0),
        'drug_b_mean_risk': encoders['mean_risk'].get(d_b, 0)
    }])
    
    score = float(ml_model.predict(features)[0])
    risk_label = "HIGH RISK" if score > 10 else "MEDIUM RISK" if score > 5 else "LOW RISK"
    
    # 2. Get Historical Evidence using DuckDB
    import duckdb
    parquet_file = os.path.join(processed_dir, "aggregated_stats.parquet").replace("\\", "/")
    query = f"""
        SELECT event, a as co_occurrences, PRR, risk_score 
        FROM read_parquet('{parquet_file}')
        WHERE drug_a = '{d_a}' AND drug_b = '{d_b}'
        ORDER BY risk_score DESC
        LIMIT 10
    """
    
    try:
        evidence_df = duckdb.query(query).df()
        evidence = evidence_df.to_dict(orient="records")
    except Exception as e:
        evidence = []
        print(f"Error querying duckdb: {e}")
        
    return {
        "drug_a": d_a,
        "drug_b": d_b,
        "ai_prediction": {
            "score": round(score, 2),
            "label": risk_label
        },
        "historical_evidence": evidence
    }
