import tensorflow as tf
from transformers import AutoTokenizer
import joblib
from app.config import MODEL_PATH

modelo = tf.keras.models.load_model(MODEL_PATH)  # ✅ FUNCIONA CON .save()
tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/tokenizer")
label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")
