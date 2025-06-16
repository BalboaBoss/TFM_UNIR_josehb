from fastapi import APIRouter
from app.schemas.ticket import TicketRequest, TicketResponse
from app.modelos.prediccion import predict_categoria

router = APIRouter()

@router.post("/clasifica", response_model=TicketResponse)
def classify_ticket(ticket: TicketRequest):
    category_id, confidence = predict_categoria(ticket.subject + " " + ticket.body)
    return TicketResponse(category=category_id, confidence=confidence)