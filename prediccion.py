import joblib
from transformers import BertTokenizerFast
import tensorflow as tf

# Rutas
MODEL_PATH = "modelos2/modelo_final"
TOKENIZER_PATH = f"{MODEL_PATH}/tokenizer"
LABEL_ENCODER_PATH = f"{MODEL_PATH}/label_encoder.pkl"

# Cargar modelo con firma explícita (serving_default)
model = tf.saved_model.load(MODEL_PATH)

# Cargar tokenizer y label encoder
tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
