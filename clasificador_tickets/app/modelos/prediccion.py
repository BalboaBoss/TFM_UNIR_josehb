from app.modelos.loader import model, tokenizer
from app.services.preprocesamiento import prepro_text
import torch
import torch.nn.functional as F

def predict_categoria(text: str):
    cleaned_text = prepro_text(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()
    return prediction, confidence