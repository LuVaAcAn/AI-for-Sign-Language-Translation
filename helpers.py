import os
import cv2
import numpy as np
import pandas as pd
from typing import NamedTuple, List
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from mediapipe.python.solutions.holistic import FACEMESH_CONTOURS, POSE_CONNECTIONS, HAND_CONNECTIONS
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

def detectarMediapipe(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    resultados = modelo.process(imagen)
    imagen.flags.writeable = True
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    return imagen, resultados

def existeCarpeta(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Carpeta creada: {path}")
    else:
        print(f"Carpeta ya existe: {path}")


def hayManos(landmarks):
    # Verifica si hay landmarks en la lista
    return len(landmarks) > 0

def listarAcciones(directorio):
    return [os.path.splitext(archivo)[0] for archivo in os.listdir(directorio) if archivo.endswith('.h5')]

def configurarCamara(camara, ancho=1280, alto=720):
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)

def dibujarKeypoints(image, landmarks):
    for hand_landmarks in landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(150, 50, 50), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(150, 100, 100), thickness=1, circle_radius=1)
        )

def guardarImagenes(frames, output_folder):
    for i, frame in enumerate(frames):
        output_path = os.path.join(output_folder, f"{i + 1}.jpg")
        try:
            print(f"Guardando frame {i + 1}: Shape={frame.shape}, Dtype={frame.dtype}, Path={output_path}")
            if not cv2.imwrite(output_path, frame):
                raise Exception("cv2.imwrite() failed")
        except Exception as e:
            print(f"Error al guardar el frame en {output_path}: {e}")
            # Probar guardarlo en una ruta diferente temporalmente
            temp_output_path = os.path.join("C:/temp_images", f"{i + 1}.jpg")
            try:
                os.makedirs("C:/temp_images", exist_ok=True)
                if not cv2.imwrite(temp_output_path, frame):
                    raise Exception("cv2.imwrite() failed en ruta temporal")
                else:
                    print(f"Frame guardado correctamente en la ruta temporal: {temp_output_path}")
            except Exception as temp_e:
                print(f"Error al guardar el frame en la ruta temporal: {temp_e}")

def extraerKeypoints(resultados):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in resultados.pose_landmarks.landmark]).flatten() if resultados.pose_landmarks else np.zeros(33 * 4)
    cara = np.array([[res.x, res.y, res.z] for res in resultados.face_landmarks.landmark]).flatten() if resultados.face_landmarks else np.zeros(468 * 3)
    mano_der = np.array([[res.x, res.y, res.z] for res in resultados.right_hand_landmarks.landmark]).flatten() if resultados.right_hand_landmarks else np.zeros(21 * 3)
    mano_izq = np.array([[res.x, res.y, res.z] for res in resultados.left_hand_landmarks.landmark]).flatten() if resultados.left_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, cara, mano_der, mano_izq])

def generaSecuencia(modelo, outputfolder):
    secuencia = []
    for imgNombre in os.listdir(outputfolder):
        rutaImg = os.path.join(outputfolder, imgNombre)
        frame = cv2.imread(rutaImg)
        _, resultados = detectarMediapipe(frame, modelo)
        puntos = extraerKeypoints(resultados)
        secuencia.append(puntos)
    return np.array(secuencia)

def agregarPuntos(df: pd.DataFrame, idMuestra: int, secuencia):
    new_rows = []
    for indiceFrame, puntos in enumerate(secuencia):
        new_rows.append({'muestra': idMuestra, 'frame': indiceFrame + 1, 'puntos_clave': puntos})
    
    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    return df

def cargarTags(acciones: List[str], directorioDatos: str):
    secuencias, etiquetas = [], []
    for etiqueta, accion in enumerate(acciones):
        hdf = os.path.join(directorioDatos, f"{accion}.h5")
        datos = pd.read_hdf(hdf, key='data')
        for _, datosMuestra in datos.groupby('muestra'):
            secuencias.append([fila['puntos_clave'] for _, fila in datosMuestra.iterrows()])
            etiquetas.append(etiqueta)
    return secuencias, etiquetas

def guardarTexto(fname: str, contenido: str):
    with open(fname, 'w') as archivo:
        archivo.write(contenido)

def actualizarOraciones(reps, oracion, oraciones):
    if len(oraciones) > 1 and oracion in oraciones[1]:
        reps += 1
        oraciones.pop(0)
        oraciones[0] = f"{oracion} (x{reps})"
    else:
        reps = 1
    return oraciones, reps

def verificar_permisos(path):
    if os.access(path, os.W_OK):
        print(f"Permisos de escritura verificados en {path}")
        return True
    else:
        print(f"Sin permisos de escritura en {path}")
        return False