import cv2
import os
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
from PIL import Image, ImageTk

dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Registro de Usuario")
        
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)
        
        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_start = tk.Button(root, text="Iniciar Registro", width=15, command=self.start_registration)
        self.btn_start.pack()

        self.user_id = ""

        self.update()

    def start_registration(self):
        self.user_id = simpledialog.askstring("ID de Usuario", "Ingresa tu ID de usuario:")
        if not self.user_id:
            messagebox.showerror("Error", "No se ha ingresado ningún ID de usuario")
            return

        count = 0
        while count < 30:
            ret, frame = self.vid.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{dataset_folder}/user_{self.user_id}_{count + 1}.jpg", face_img)
                count += 1

            self.show_frame(frame)
            self.root.update()

        messagebox.showinfo("Registro Completo", "Imágenes guardadas correctamente")

    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.show_frame(frame)
        self.root.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()
