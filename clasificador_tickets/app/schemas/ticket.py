from pydantic import BaseModel

class TicketRequest(BaseModel):
    subject: str
    body: str

class TicketResponse(BaseModel):
    category: int
    confidence: float