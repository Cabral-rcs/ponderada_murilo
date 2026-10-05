"""Treina um modelo de regressão para prever o fechamento do BTC no dia seguinte
a partir dos últimos N_LAGS fechamentos e exporta o artefato em .pkl."""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

TICKER = "BTC-USD"
PERIOD = "2y"
N_LAGS = 7
TEST_SIZE = 0.2

DATA_PATH = "data/btc.csv"
ARTIFACTS_DIR = "artifacts"


def load_data() -> pd.Series:
    """Usa o CSV local se existir (reprodutível offline); senão baixa do Yahoo Finance."""
    if os.path.exists(DATA_PATH):
        print(f"[dados] lendo {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    else:
        import yfinance as yf

        print(f"[dados] baixando {TICKER} ({PERIOD}) do Yahoo Finance")
        raw = yf.download(TICKER, period=PERIOD, interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("Download vazio. Coloque um CSV em data/btc.csv com colunas Date,Close.")
        close = raw["Close"].squeeze()
        df = pd.DataFrame({"Date": close.index, "Close": close.values})
        df.to_csv(DATA_PATH, index=False)
        print(f"[dados] salvo em {DATA_PATH}")

    return df.sort_values("Date").set_index("Date")["Close"].dropna()


def make_lags(close: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Cada linha: fechamentos t-N..t-1 (features) -> fechamento t (alvo)."""
    features = [f"lag_{i}" for i in range(N_LAGS, 0, -1)]  # lag_7 (mais antigo) ... lag_1 (mais recente)
    X = pd.concat({name: close.shift(i) for name, i in zip(features, range(N_LAGS, 0, -1))}, axis=1)
    y = close.rename("target")
    data = pd.concat([X, y], axis=1).dropna()
    return data[features], data["target"]


def main() -> None:
    close = load_data()
    print(f"[dados] {len(close)} dias, de {close.index.min().date()} a {close.index.max().date()}")

    X, y = make_lags(close)

    # Split temporal: treina no passado, testa no futuro (sem shuffle).
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train.values, y_train.values)

    mae_model = mean_absolute_error(y_test, model.predict(X_test.values))
    mae_baseline = mean_absolute_error(y_test, X_test["lag_1"])  # amanhã = hoje

    # Modelo final treinado com todos os dados.
    model.fit(X.values, y.values)

    metrics = {
        "mae_model_usd": round(float(mae_model), 2),
        "mae_baseline_usd": round(float(mae_baseline), 2),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    artifact = {
        "model": model,
        "n_lags": N_LAGS,
        "features": list(X.columns),
        "ticker": TICKER,
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "last_closes": [round(float(v), 2) for v in close.iloc[-N_LAGS:]],
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(artifact, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump({k: v for k, v in artifact.items() if k != "model"}, f, indent=2)

    next_day = model.predict(np.array([artifact["last_closes"]]))[0]
    print(f"[treino] MAE modelo: {metrics['mae_model_usd']} USD | MAE baseline: {metrics['mae_baseline_usd']} USD")
    print(f"[treino] previsão para o próximo dia: {next_day:.2f} USD")
    print(f"[artefato] salvo em {ARTIFACTS_DIR}/model.pkl")


if __name__ == "__main__":
    main()
