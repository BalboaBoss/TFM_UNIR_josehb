import tensorflow as tf
import joblib
from transformers import AutoTokenizer
from app.config import MODEL_PATH,TOKENIZER_PATH,LABEL_ENCODER_PATH
from keras.models import load_model


# Cargar modelo entrenado (espera input_ids y attention_mask)
modelo = load_model(MODEL_PATH, compile=False) 
# Cargar tokenizer HuggingFace
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Cargar label encoder
label_encoder = joblib.load(LABEL_ENCODER_PATH)