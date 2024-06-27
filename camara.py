import os
import cv2
import numpy as np
from tts import speakText
from tensorflow.keras.models import load_model
import mediapipe as mp
from mediapipe.python.solutions.holistic import Holistic
from helpers import listarAcciones, hayManos
from constants import DATA_DIR, MAX_FRAMES, MIN_FRAMES, MODEL_DIR, MODEL_FILE, TEXT_FONT, TEXT_SIZE, TEXT_POS
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class Camara:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def capturar_puntos_clave(self, imagen):
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultados = self.holistic.process(imagen_rgb)

        if resultados.face_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
        if resultados.pose_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
        if resultados.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if resultados.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

        return imagen, resultados

    def extraer_puntos_clave(self, resultados):
        puntos_clave_manos = np.zeros((2, 21, 3))
        puntos_clave_pose = np.zeros((33, 3))
        puntos_clave_face = np.zeros((468, 3))

        if resultados.left_hand_landmarks:
            for j, lm in enumerate(resultados.left_hand_landmarks.landmark):
                puntos_clave_manos[0, j] = [lm.x, lm.y, lm.z]
        if resultados.right_hand_landmarks:
            for j, lm in enumerate(resultados.right_hand_landmarks.landmark):
                puntos_clave_manos[1, j] = [lm.x, lm.y, lm.z]

        if resultados.pose_landmarks:
            for i, lm in enumerate(resultados.pose_landmarks.landmark):
                puntos_clave_pose[i] = [lm.x, lm.y, lm.z]

        if resultados.face_landmarks:
            for i, lm in enumerate(resultados.face_landmarks.landmark):
                if i < 468:
                    puntos_clave_face[i] = [lm.x, lm.y, lm.z]

        puntos_clave = np.concatenate([puntos_clave_manos.flatten(), puntos_clave_pose.flatten(), puntos_clave_face.flatten()])
        if puntos_clave.size != 1629:
            print(f"Advertencia: La dimensión de los puntos clave es {puntos_clave.size}, pero se esperaban 1662.")

        return puntos_clave

def evaluarModelo(model, video_path, threshold=0.7):
    camara = Camara()
    actions = listarAcciones(DATA_DIR)
    secuencia, oracion = [], []
    iframe = 0

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}.")
        return

    print("Procesando video...")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame, resultados = camara.capturar_puntos_clave(frame)
        keypoints = camara.extraer_puntos_clave(resultados)
        secuencia.append(keypoints)

        if len(secuencia) >= MIN_FRAMES:
            secuencia_array = np.array(secuencia[-MAX_FRAMES:])
            print(f"Secuencia shape: {secuencia_array.shape}")
            try:
                res = model.predict(np.expand_dims(secuencia_array, axis=0))[0]
                prediccion = actions[np.argmax(res)]
                print(f"Predicción del modelo: {prediccion} (confianza: {res[np.argmax(res)]})")
                if res[np.argmax(res)] > threshold:
                    sent = prediccion
                    oracion.insert(0, sent)
                    print(f"Traducción detectada: {sent}")
                    speakText(sent)
            except Exception as e:
                print(f"Error al predecir: {e}")
            secuencia = []

        cv2.rectangle(frame, (0, 0), (640, 35), (245, 117, 16), -1)
        cv2.putText(frame, ' |=| '.join(oracion), TEXT_POS, TEXT_FONT, TEXT_SIZE, (255, 255, 255))
        cv2.imshow('TRADUCCION', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    print("Procesamiento del video completado.")

if __name__ == "__main__":
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    print(f"Cargando modelo desde: {model_path}")
    lstm_model = load_model(model_path)
    video_path = "Prueba.mp4"
    evaluarModelo(lstm_model, video_path)