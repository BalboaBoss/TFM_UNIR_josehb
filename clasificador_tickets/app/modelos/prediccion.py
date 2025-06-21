from app.modelos.loader import modelo, tokenizer, label_encoder
import tensorflow as tf
import numpy as np

def predict_categoria(texto: str) -> str:
    inputs = tokenizer(texto, return_tensors="tf", padding=True, truncation=True, max_length=512)

    # Pasar como lista en el orden esperado por el modelo
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]


    outputs = modelo([input_ids, attention_mask])


    probs = tf.nn.softmax(outputs, axis=-1).numpy()

    predicted_index = np.argmax(probs, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_label

