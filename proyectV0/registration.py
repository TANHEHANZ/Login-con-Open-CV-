import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
from video_feed import VideoFeed
from model import update_model

class RegistrationScreen:
    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.root.title("Registro de Usuario")

        self.video_source = 0
        self.vid = VideoFeed(self.video_source, self.root)

        self.canvas = self.vid.canvas
        self.canvas.pack()

        self.btn_start = tk.Button(root, text="Iniciar Registro", width=20, command=self.start_registration)
        self.btn_start.pack()

        self.btn_back = tk.Button(root, text="Volver", width=20, command=self.back_to_menu)
        self.btn_back.pack(pady=10)

        self.user_id = ""
        self.update()

    def start_registration(self):
        self.user_id = simpledialog.askstring("ID de Usuario", "Ingresa tu ID de usuario:")
        if not self.user_id:
            messagebox.showerror("Error", "No se ha ingresado ningún ID de usuario")
            return

        count = 0
        while count < 30:
            ret, frame = self.vid.get_frame()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"dataset/user_{self.user_id}_{count + 1}.jpg", face_img)
                count += 1

            self.vid.show_frame(frame)
            self.root.update()

        messagebox.showinfo("Registro Completo", "Imágenes guardadas correctamente")
        update_model()
        self.back_to_menu()

    def update(self):
        self.vid.update()

    def back_to_menu(self):
        self.vid.release()
        self.canvas.pack_forget()
        self.btn_start.pack_forget()
        self.btn_back.pack_forget()
        self.parent.show_main_menu()
