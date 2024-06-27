import os
import cv2
#RUTAS/DIRECCIONES
ROOT_DIR = os.getcwd()
FRAME_DIR = os.path.join(ROOT_DIR, "acciones")
DATA_DIR = os.path.join(ROOT_DIR, "datos")
MODEL_DIR = os.path.join(ROOT_DIR, "modelos")
#CONSTANTES
MAX_FRAMES = 30
KEYPOINT_COUNT = 1662
MIN_FRAMES = 2
MODEL_FILE = f"modelo_{MAX_FRAMES}.keras" 
#FUENTE DE LETRA
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 1
TEXT_POS = (2, 30)