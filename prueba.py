from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import joblib
from transformers import BertTokenizerFast

app = FastAPI()

# Rutas
MODEL_PATH = "modelos3/modelo_final"
TOKENIZER_PATH = f"{MODEL_PATH}/tokenizer"
LABEL_ENCODER_PATH = f"{MODEL_PATH}/label_encoder.pkl"

# Cargar modelo con firma explícita (serving_default)
model = tf.saved_model.load(MODEL_PATH)

# Cargar tokenizer y label encoder
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Modelo de entrada para la API
class TextInput(BaseModel):
    text: str

# Endpoint de predicción
@app.post("/predict")
def predict(input: TextInput):
    try:
        # Tokenizar el texto
        tokens = tokenizer(
            input.text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="tf"
        )

        # Preparar input como diccionario
        input_dict = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

        # Predecir
        predict_fn = model.signatures["serving_default"]
        predictions = predict_fn(input_dict)["output_0"]
        predicted_index = tf.argmax(predictions, axis=1).numpy()[0]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        return {"label": predicted_label}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
