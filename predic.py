import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Configuración
dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

def load_label_ids():
    label_ids = {}
    with open('label_ids.txt', 'r') as f:
        for line in f:
            nombre, idx = line.strip().split(':')
            label_ids[int(idx)] = nombre
    return label_ids

def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('modelo_LBPHV2.yml')

    # Cargar etiquetas y nombres
    label_ids = load_label_ids()

    # Iniciar la cámara
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No se pudo capturar el video")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            rostro_recortado = gray_frame[y:y+h, x:x+w]
            label_predicho, confianza = recognizer.predict(rostro_recortado)
            print(f"Confianza: {confianza}, Label: {label_predicho}")

            # Si la confianza es suficientemente alta (ajusta este valor según sea necesario)
            if confianza < 100:
                nombre_predicho = label_ids.get(label_predicho, "Desconocido")
                print(f"Persona reconocida: {nombre_predicho} con confianza {confianza}")

                # Cerrar la cámara y mostrar la información del usuario
                video_capture.release()
                cv2.destroyAllWindows()
                mostrar_foto_usuario(nombre_predicho)
                return  # Terminar el bucle

            else:
                print("No se reconoció el rostro.")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Desconocido", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def mostrar_foto_usuario(nombre_usuario):
    # Crear una ventana nueva para mostrar la imagen y el nombre
    window = tk.Tk()
    window.title("Usuario Reconocido")

    # Buscar y cargar la imagen del usuario
    img_path = None
    for filename in os.listdir(dataset_folder):
        if filename.startswith(f"user_{nombre_usuario}"):
            img_path = os.path.join(dataset_folder, filename)
            break

    if img_path:
        img = Image.open(img_path)
        img = img.resize((300, 300))  # Redimensionar la imagen para que quepa en la ventana
        img_tk = ImageTk.PhotoImage(img)

        # Crear etiquetas para mostrar la imagen y el nombre
        label_img = tk.Label(window, image=img_tk)
        label_img.pack()

        label_name = tk.Label(window, text=f"Nombre: {nombre_usuario}", font=("Helvetica", 16))
        label_name.pack()

        window.mainloop()
    else:
        print(f"No se encontró la imagen de {nombre_usuario}")

if __name__ == "__main__":
    recognize_face()
