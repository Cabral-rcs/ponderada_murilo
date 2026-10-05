"""Register: módulo Python da API que guarda as entradas e as predições em CSV."""

import csv
import os
import uuid
from datetime import datetime, timezone

CSV_PATH = os.getenv("REGISTRO_PATH", "registros/registro.csv")
CAMPOS = ["id", "timestamp", "closes", "prediction", "status"]


def _ler() -> list[dict]:
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _escrever(linhas: list[dict]) -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerows(linhas)


def registrar_entrada(closes: list[float]) -> str:
    """Salva os dados recebidos do frontend. Retorna o id do registro."""
    registro_id = uuid.uuid4().hex[:8]
    linhas = _ler()
    linhas.append({
        "id": registro_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "closes": ";".join(str(c) for c in closes),
        "prediction": "",
        "status": "recebido",
    })
    _escrever(linhas)
    return registro_id


def registrar_predicao(registro_id: str, prediction: float | None, status: str) -> None:
    """Atualiza o registro com a predição devolvida pelo modelo (ou o erro)."""
    linhas = _ler()
    for linha in linhas:
        if linha["id"] == registro_id:
            linha["prediction"] = "" if prediction is None else prediction
            linha["status"] = status
    _escrever(linhas)


def listar() -> list[dict]:
    return _ler()
