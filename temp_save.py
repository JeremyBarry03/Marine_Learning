import tensorflow as tf
from tensorflow.keras import layers, models
from src.keras_utils import ensure_preprocess_lambda_deserializable
image_size = 384
inputs = layers.Input(shape=(image_size, image_size, 3), name='input_image')
model = models.Model(inputs=inputs, outputs=inputs)
try:
    model.save('temp_lambda_model.keras')
except Exception as exc:
    print('save error:', exc)
else:
    print('saved ok')
