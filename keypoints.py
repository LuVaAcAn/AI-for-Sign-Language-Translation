import os
import pandas as pd
from mediapipe.python.solutions.holistic import Holistic
from helpers import generaSecuencia, agregarPuntos
from constants import DATA_DIR, FRAME_DIR, ROOT_DIR

def crearKeypoints(frames_path, save_path):
    data = pd.DataFrame([])
    with Holistic() as heuristica:
        for imuestras, nommuestras in enumerate(os.listdir(frames_path), 1):
            muestra_dir = os.path.join(frames_path, nommuestras)
            secuencia = generaSecuencia(heuristica, muestra_dir)
            data = agregarPuntos(data, imuestras, secuencia)
    data.to_hdf(save_path, key="data", mode="w")

if __name__ == "__main__":
    palabras_dir = os.path.join(ROOT_DIR, FRAME_DIR)
    for nompalabra in os.listdir(palabras_dir):
        palabra_dir = os.path.join(palabras_dir, nompalabra)
        hdf_dir = os.path.join(DATA_DIR, f"{nompalabra}.h5")
        print(f'KEYPOINTS DE: "{nompalabra}"')
        crearKeypoints(palabra_dir, hdf_dir)
        print(f"¡LISTO!")