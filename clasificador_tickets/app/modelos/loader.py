from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.config import MODEL_PATH

modelo = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)