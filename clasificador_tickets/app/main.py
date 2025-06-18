# ticket_classifier_api/app/main.py
from fastapi import FastAPI
from app.routes.clasificador import router as clasificador_router

app = FastAPI(title="Clasificador de tickets")

app.include_router(clasificador_router, prefix="/api")