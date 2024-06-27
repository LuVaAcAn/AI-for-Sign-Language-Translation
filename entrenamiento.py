import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from modelo import NUM_EPOCH, modelo
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from helpers import listarAcciones, cargarTags
from constants import MAX_FRAMES, MODEL_FILE, KEYPOINT_COUNT
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def entrenaModelo(data_path, model_path):
    actions = listarAcciones(data_path)
    sequences, labels = cargarTags(actions, data_path)
    sequences = pad_sequences(sequences, maxlen=MAX_FRAMES, padding='post', truncating='post', dtype='float32')
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    
    print(f"Dimensiones de X: {X.shape}")  # Añadir esta línea para verificar las dimensiones

    model = modelo(len(actions))
    model.fit(X, y, epochs=NUM_EPOCH)
    model.summary()
    model.save(model_path)

if __name__ == "__main__":
    root = os.getcwd()
    data_path = os.path.join(root, "datos")
    save_path = os.path.join(root, "modelos")
    model_path = os.path.join(save_path, MODEL_FILE)
    entrenaModelo(data_path, model_path)