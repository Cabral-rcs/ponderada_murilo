"""API: recebe a requisição do frontend, registra no CSV, verifica o modelo
(GET /health) e, se ok, pede a predição (POST /prever)."""

import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import register

MODELO_URL = os.getenv("MODELO_URL", "http://modelo:8000")

app = FastAPI(title="API BTC")


class PredicaoRequest(BaseModel):
    closes: list[float]


@app.get("/health")
def health() -> dict:
    """Status da API e do modelo (o frontend usa para mostrar os últimos fechamentos)."""
    try:
        modelo = httpx.get(f"{MODELO_URL}/health", timeout=5).json()
    except httpx.HTTPError:
        modelo = {"status": "indisponivel"}
    return {"status": "ok", "modelo": modelo}


@app.post("/predicao")
def predicao(req: PredicaoRequest) -> dict:
    # 1. Register salva os dados
    try:
        registro_id = register.registrar_entrada(req.closes)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"erro ao registrar: {e}")

    # 2. Verifica se o modelo está ativo
    try:
        httpx.get(f"{MODELO_URL}/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        register.registrar_predicao(registro_id, None, "modelo_indisponivel")
        raise HTTPException(status_code=503, detail="modelo indisponível")

    # 3. Pede a predição
    resp = httpx.post(f"{MODELO_URL}/prever", json={"closes": req.closes}, timeout=10)
    if resp.status_code != 200:
        register.registrar_predicao(registro_id, None, "erro_modelo")
        raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail"))

    # 4. Register salva a predição e a API devolve ao frontend
    prediction = resp.json()["prediction"]
    register.registrar_predicao(registro_id, prediction, "ok")
    return {"id": registro_id, "prediction": prediction}


@app.get("/historico")
def historico() -> list[dict]:
    return register.listar()
