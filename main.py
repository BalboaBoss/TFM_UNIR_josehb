from fastapi import FastAPI
from prediccion import model, tokenizer, label_encoder
from pydantic import BaseModel
from fastapi import HTTPException
import tensorflow as tf
import time

app = FastAPI()

# Modelo de entrada para la API
class TextInput(BaseModel):
    subject: str
    body: str

# Endpoint de predicción
@app.post("/predict")
def predict(input: TextInput):
    try:
        full_text = f"{input.subject} {input.body}"

        tokens = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="tf"
        )

        # CAMBIO CLAVE: usar nombres reales de los tensores
        inputs = {
            "inputs": tokens["input_ids"],
            "inputs_1": tokens["attention_mask"]
        }

        predictions = model.signatures["serving_default"](**inputs)["output_0"]
        predicted_index = tf.argmax(predictions, axis=1).numpy()[0]
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        duration = time.time() - start
        print(f"Tiempo de predicción: {duration:.4f} segundos")
        return {"label": predicted_label}

    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=str(e))
