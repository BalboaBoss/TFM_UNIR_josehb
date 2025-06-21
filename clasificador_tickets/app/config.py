import os

MODEL_PATH = os.getenv("MODEL_PATH", "modelos/modelo_final")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "modelos/modelo_final/tokenizer")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "modelo_bert_guardado/label_encoder.pkl")
THRESHOLD = 0.5