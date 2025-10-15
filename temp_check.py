import tensorflow as tf
from src.keras_utils import ensure_preprocess_lambda_deserializable, restore_preprocess_lambda
from src.config import MARINE_CONFIG
ensure_preprocess_lambda_deserializable((384, 384, 3))
model = tf.keras.models.load_model('models/marine_classifier.keras', safe_mode=False)
restore_preprocess_lambda(model, MARINE_CONFIG['model']['backbone'])
print('Trainable params:', model.count_params())
