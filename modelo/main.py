"""Container de inferência: carrega o artefato gerado pelo treinamento e expõe
GET /health (serviço ativo?) e POST /prever (predição do próximo fechamento)."""

import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")

app = FastAPI(title="Modelo BTC")
artifact: dict | None = None


def load_artifact() -> None:
    global artifact
    if os.path.exists(MODEL_PATH):
        artifact = joblib.load(MODEL_PATH)
        print(f"[modelo] artefato carregado de {MODEL_PATH} (treinado em {artifact['trained_at']})")
    else:
        print(f"[modelo] artefato não encontrado em {MODEL_PATH}")


@app.on_event("startup")
def startup() -> None:
    load_artifact()


class PredictRequest(BaseModel):
    closes: list[float]  # últimos N fechamentos, do mais antigo para o mais recente


@app.get("/health")
def health() -> dict:
    if artifact is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")
    return {
        "status": "ok",
        "ticker": artifact["ticker"],
        "n_lags": artifact["n_lags"],
        "trained_at": artifact["trained_at"],
        "sklearn_version": artifact["sklearn_version"],
        "metrics": artifact["metrics"],
        "last_closes": artifact["last_closes"],
    }


@app.post("/prever")
def prever(req: PredictRequest) -> dict:
    if artifact is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")
    n_lags = artifact["n_lags"]
    if len(req.closes) != n_lags:
        raise HTTPException(status_code=422, detail=f"envie exatamente {n_lags} fechamentos, recebido {len(req.closes)}")
    prediction = artifact["model"].predict(np.array([req.closes]))[0]
    return {"prediction": round(float(prediction), 2), "input": req.closes, "trained_at": artifact["trained_at"]}
