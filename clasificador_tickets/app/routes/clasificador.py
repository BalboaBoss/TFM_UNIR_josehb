from fastapi import APIRouter
from pydantic import BaseModel
from app.modelos.prediccion import predict_categoria

router = APIRouter()

class Ticket(BaseModel):
    texto: str

@router.post("/clasificar")
def clasificar_ticket(ticket: Ticket):
    prediccion = predict_categoria(ticket.texto)
    return {"categoria": prediccion}