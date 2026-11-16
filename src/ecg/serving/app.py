"""API phục vụ dự đoán ECG (FastAPI).

Tuần 5 chỉ có GET /health để kiểm tra image serving chạy được. POST /predict (nhận
file WFDB .dat + .hea, tiền xử lý bằng preprocess_ecg, nạp model từ MLflow Model
Registry) làm ở tuần 7.

Chạy cục bộ (từ gốc repo, venv đã kích hoạt):
    PYTHONPATH=src uvicorn ecg.serving.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="ECG superdiagnostic API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
