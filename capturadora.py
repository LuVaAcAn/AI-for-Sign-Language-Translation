import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.holistic import Holistic
from helpers import detectarMediapipe, existeCarpeta, verificar_permisos, dibujarKeypoints, hayManos, guardarImagenes
from constants import ROOT_DIR, FRAME_DIR, TEXT_FONT, TEXT_POS, TEXT_SIZE, KEYPOINT_COUNT

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

    def capturar_manos(self, imagen):
        imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        resultados = self.holistic.process(imagen_rgb)

        # Dibujar puntos clave
        if resultados.face_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.face_landmarks, mp.solutions.holistic.FACEMESH_CONTOURS)
        if resultados.pose_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
        if resultados.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        if resultados.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(imagen, resultados.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

        return imagen, resultados

    def extraer_puntos_clave(self, landmarks):
     puntos_clave = np.zeros((2, 21, 3))  # 2 manos, 21 puntos por mano, 3 coordenadas por punto (x, y, z)
     for i, hand_landmarks in enumerate(landmarks):
        if i < 2:
            for j, lm in enumerate(hand_landmarks.landmark):
                puntos_clave[i, j] = [lm.x, lm.y, lm.z]
     puntos_clave = puntos_clave.flatten()
    
    # Ajustar tamaño de los puntos clave
     if puntos_clave.size != KEYPOINT_COUNT:
        print(f"Advertencia: La dimensión de los puntos clave es {puntos_clave.size}, pero se esperaban {KEYPOINT_COUNT}.")
    
     return puntos_clave

def procesar_video(video_path, output_path, margfr=2, minfr=8):
    camara = Camara()
    existeCarpeta(output_path)
    if not verificar_permisos(output_path):
        output_path = "C:/temp_images"
        existeCarpeta(output_path)
        
    cant_muestras = len(os.listdir(output_path))
    isample = 0
    iframe = 0
    frames = []

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}.")
        return

    while True:
        ret, frame = video.read()
        if not ret:
            print(f"Error: No se pudo leer el frame del video {video_path}.")
            if len(frames) > minfr:
                output_folder = os.path.join(output_path, f"sample_{cant_muestras + isample + 1}")
                existeCarpeta(output_folder)
                guardarImagenes(frames, output_folder)
                print(f"Guardando {len(frames)} frames en {output_folder}")
            break

        frame, resultados = camara.capturar_manos(frame)
        landmarks = []
        if resultados.left_hand_landmarks:
            landmarks.append(resultados.left_hand_landmarks)
        if resultados.right_hand_landmarks:
            landmarks.append(resultados.right_hand_landmarks)
        
        if hayManos(landmarks):
            iframe += 1
            print(f"Manos detectadas: Frame {iframe}")
            if iframe > margfr:
                cv2.putText(frame, '=CAPTURANDO FRAMES=', TEXT_POS, TEXT_FONT, TEXT_SIZE, (255, 50, 0))
                frames.append(np.asarray(frame))
                print(f"Capturando frame {len(frames)}")
        else:
            if len(frames) > minfr:
                frames = frames[:-margfr]
                output_folder = os.path.join(output_path, f"sample_{cant_muestras + isample + 1}")
                existeCarpeta(output_folder)
                guardarImagenes(frames, output_folder)
                print(f"Guardando {len(frames)} frames en {output_folder}")
                isample += 1
                frames = []
            iframe = 0
            cv2.putText(frame, '[PON TUS MANOS]', TEXT_POS, TEXT_FONT, TEXT_SIZE, (0, 220, 100))
            print("Esperando manos")

        dibujarKeypoints(frame, landmarks)
        cv2.imshow(f'MUESTRAS DE "{os.path.basename(output_path)}"', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    if len(frames) > minfr:
        frames = frames[:-margfr] if len(frames) > margfr else frames
        output_folder = os.path.join(output_path, f"sample_{cant_muestras + isample + 1}")
        existeCarpeta(output_folder)
        guardarImagenes(frames, output_folder)
        print(f"Guardando {len(frames)} frames en {output_folder}")

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    carpeta_videos = "videos"
    carpeta_salida = os.path.join(ROOT_DIR, FRAME_DIR)

    for video_file in os.listdir(carpeta_videos):
        video_path = os.path.join(carpeta_videos, video_file)
        output_path = os.path.join(carpeta_salida, os.path.splitext(video_file)[0])
        print(f"Procesando video {video_path}")
        procesar_video(video_path, output_path)
        print(f"Finalizado procesamiento de {video_path}")