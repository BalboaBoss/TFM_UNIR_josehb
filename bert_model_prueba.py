import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import joblib
import os
import kagglehub

# 1. Descargar dataset
path = kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets")
df = pd.read_csv(f"{path}/dataset-tickets-multi-lang3-4k.csv")

# 2. Preparar texto y etiquetas
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
df = df[df["language"].isin(["en", "es", "de"])]
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["queue"])

# 3. División y sobremuestreo
X_train, X_val, y_train, y_val = train_test_split(df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(np.array(X_train).reshape(-1, 1), y_train)
X_train = X_train.ravel()

# 4. Tokenización
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(texts):
    return tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="tf")

train_enc = tokenize(X_train)
val_enc = tokenize(X_val)

# 5. Modelo BERT con entrada tipo diccionario
num_labels = len(label_encoder.classes_)
bert = TFAutoModel.from_pretrained(model_name)

# Inputs como diccionario (clave para servirlo fácil luego)
inputs = {
    "input_ids": tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids"),
    "attention_mask": tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
}

embedding = bert(inputs["input_ids"], attention_mask=inputs["attention_mask"])[1]  # pooled_output
output = tf.keras.layers.Dense(num_labels, activation="softmax")(embedding)

model = tf.keras.Model(inputs=inputs, outputs=output)

# Compilar y entrenar
model.compile(optimizer=tf.keras.optimizers.Adam(2e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(
    x={"input_ids": train_enc["input_ids"], "attention_mask": train_enc["attention_mask"]},
    y=y_train,
    validation_data=(
        {"input_ids": val_enc["input_ids"], "attention_mask": val_enc["attention_mask"]},
        y_val
    ),
    epochs=5,
    batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)]
)

# 6. Guardar modelo con firma explícita para servir como diccionario
@tf.function(input_signature=[{
    "input_ids": tf.TensorSpec([None, 128], tf.int32),
    "attention_mask": tf.TensorSpec([None, 128], tf.int32)
}])
def serving_fn(inputs):
    return model(inputs)

export_path = "modelos3/modelo_final"
model.save(export_path, include_optimizer=False, signatures={"serving_default": serving_fn})
tokenizer.save_pretrained(f"{export_path}/tokenizer")
joblib.dump(label_encoder, f"{export_path}/label_encoder.pkl")
