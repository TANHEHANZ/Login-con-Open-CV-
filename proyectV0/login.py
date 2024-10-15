import tkinter as tk
from tkinter import messagebox
import cv2
from video_feed import VideoFeed
from model import load_label_ids

class LoginScreen:
    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.root.title("Iniciar Sesión")

        self.video_source = 0
        self.vid = VideoFeed(self.video_source, self.root)

        self.canvas = self.vid.canvas
        self.canvas.pack()

        self.btn_back = tk.Button(root, text="Volver", width=20, command=self.back_to_menu)
        self.btn_back.pack(pady=10)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('modelo_LBPHV2.yml')
        self.label_ids = load_label_ids()

        self.face_detected = False
        self.check_login()

    def check_login(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while not self.face_detected:
            ret, frame = self.vid.get_frame()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                rostro_recortado = gray_frame[y:y+h, x:x+w]
                label_predicho, confianza = self.recognizer.predict(rostro_recortado)

                if confianza < 100:
                    nombre_predicho = self.label_ids.get(label_predicho, "Desconocido")
                    self.face_detected = True
                    self.show_login_success(nombre_predicho)
                    return

            self.vid.show_frame(frame)
            self.root.update()

    def show_login_success(self, nombre):
        messagebox.showinfo("Inicio de Sesión", f"Has iniciado sesión como {nombre}")

    def back_to_menu(self):
        self.vid.release()
        self.canvas.pack_forget()
        self.btn_back.pack_forget()
        self.parent.show_main_menu()
