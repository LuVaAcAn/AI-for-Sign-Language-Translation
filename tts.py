from gtts import gTTS
from time import sleep
import pygame as pg
import os

def speakText(text):
    tts = gTTS(text=text, lang='es')
    audioFile = "senias.mp3"
    tts.save(audioFile)
    
    pg.init()
    pg.mixer.init()
    pg.mixer.music.load(audioFile)
    pg.mixer.music.play()
    
    while pg.mixer.music.get_busy():
        sleep(1)

    pg.mixer.quit()
    pg.quit()
    os.remove(audioFile)

if __name__ == "__main__":
    msg = "pruebas"
    speakText(msg)