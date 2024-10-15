import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# Configuración
dataset_folder = "dataset"
model_file = "modelo_LBPHV2.yml"
labels_file = "label_ids.txt"

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("400x300") 
        self.root.title("Sistema de Reconocimiento Facial")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack()

        self.btn_register = tk.Button(self.main_frame, text="Registrarse", width=40, command=self.open_registration)
        self.btn_register.pack(pady=10)

        self.btn_login = tk.Button(self.main_frame, text="Iniciar Sesión", width=40, command=self.open_login)
        self.btn_login.pack(pady=10)

    def open_registration(self):
        self.main_frame.pack_forget()
        self.root.geometry("800x600")
        RegistrationScreen(self.root, self)

    def open_login(self):
        self.main_frame.pack_forget()
        self.root.geometry("800x600")
        LoginScreen(self.root, self)

    def show_main_menu(self):
        self.main_frame.pack()
        self.root.geometry("400x300")


class RegistrationScreen:
    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.root.title("Registro de Usuario")
        
        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_start = tk.Button(root, text="Iniciar Registro", width=20, command=self.start_registration)
        self.btn_start.pack()

        self.btn_back = tk.Button(root, text="Volver", width=20, command=self.back_to_menu)
        self.btn_back.pack(pady=10)

        self.user_id = ""
        self.update()

    def start_registration(self):
        # Solicitar ID de usuario
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
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Dibujar rectángulo mientras se captura
                face_img = gray[y:y+h, x:x+w]
                cv2.imwrite(f"{dataset_folder}/user_{self.user_id}_{count + 1}.jpg", face_img)
                count += 1

            self.show_frame(frame)
            self.root.update()

        # Mostrar mensaje de éxito
        messagebox.showinfo("Registro Completo", "Imágenes guardadas correctamente")
        self.update_model()
        self.back_to_menu()

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

    def back_to_menu(self):
        self.vid.release()
        self.canvas.pack_forget()
        self.btn_start.pack_forget()
        self.btn_back.pack_forget()
        self.parent.show_main_menu()

    def update_model(self):
        image_paths = []
        label_ids = {}
        current_id = 0
        x_train = []
        y_labels = []
        for file in os.listdir(dataset_folder):
         if file.endswith("jpg"):
            image_path = os.path.join(dataset_folder, file)

            # Asegurarse de que el nombre del archivo tenga el formato correcto
            parts = file.split("_")
            if len(parts) < 2:
                print(f"El nombre del archivo '{file}' no tiene el formato esperado.")
                continue
            
            label = parts[1]  # Extraer ID de usuario del nombre del archivo

            # Asignar un ID a cada etiqueta de usuario
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            # Cargar la imagen en escala de grises
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            x_train.append(image)
            y_labels.append(label_ids[label])

    # Guardar las etiquetas en un archivo para su uso posterior
        with open(labels_file, 'w') as f:
         for label, id_ in label_ids.items():
            f.write(f"{label}:{id_}\n")

    # Entrenar el modelo de reconocimiento facial con LBPH
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(x_train, np.array(y_labels))
        recognizer.save(model_file)

        messagebox.showinfo("Modelo Actualizado", "El modelo de reconocimiento facial ha sido actualizado.")


class LoginScreen:
    def __init__(self, root, parent):
        self.root = root
        self.parent = parent
        self.root.title("Iniciar Sesión")

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_back = tk.Button(root, text="Volver", width=20, command=self.back_to_menu)
        self.btn_back.pack(pady=10)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('modelo_LBPHV2.yml')
        self.label_ids = self.load_label_ids()

        self.face_detected = False  # Agregado: bandera para saber si se ha detectado un rostro exitosamente
        self.check_login()

    def load_label_ids(self):
        label_ids = {}
        with open('label_ids.txt', 'r') as f:
            for line in f:
                nombre, idx = line.strip().split(':')
                label_ids[int(idx)] = nombre
        return label_ids

    def check_login(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while not self.face_detected:  # Agregado: bucle hasta que se detecte un rostro exitosamente
            ret, frame = self.vid.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in face_locations:
                rostro_recortado = gray_frame[y:y+h, x:x+w]
                label_predicho, confianza = self.recognizer.predict(rostro_recortado)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Remarcar la cara mientras busca

                if confianza < 100:  # Si la confianza es aceptable, procedemos
                    nombre_predicho = self.label_ids.get(label_predicho, "Desconocido")
                    self.face_detected = True  # Se detectó un rostro correctamente
                    self.show_login_success(nombre_predicho)
                    return  # Terminar la función cuando se encuentre la persona
                else:
                    # Aquí no hacemos nada, simplemente continuamos en el bucle
                    pass

            self.show_frame(frame)
            self.root.update()

    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

    def show_login_success(self, nombre):
        messagebox.showinfo("Inicio de Sesión", f"Has iniciado sesión como {nombre}")
        self.display_user_image(nombre)

    def display_user_image(self, nombre_usuario):
        self.canvas.pack_forget()
        self.btn_back.pack_forget()

        img_path = None
        for filename in os.listdir(dataset_folder):
            if filename.startswith(f"user_{nombre_usuario}"):
                img_path = os.path.join(dataset_folder, filename)
                break

        if img_path:
            img = Image.open(img_path)
            img = img.resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)

            self.label_img = tk.Label(self.root, image=img_tk)
            self.label_img.image = img_tk
            self.label_img.pack()

            self.label_name = tk.Label(self.root, text=f"Nombre: {nombre_usuario}", font=("Helvetica", 16))
            self.label_name.pack()

        self.btn_back_to_menu = tk.Button(self.root, text="Volver al Menú", width=20, command=self.back_to_menu)
        self.btn_back_to_menu.pack(pady=10)

    def back_to_menu(self):
        self.vid.release()
        if hasattr(self, 'label_img'):
            self.label_img.pack_forget()
        if hasattr(self, 'label_name'):
            self.label_name.pack_forget()
        self.canvas.pack_forget()
        self.btn_back_to_menu.pack_forget()
        self.parent.show_main_menu()



if __name__ == '__main__':
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()
