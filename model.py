

import cv2
import os
import numpy as np

dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    rostros, nombres = Img(dataset_folder)

    if not rostros:
        print("No se encontraron imágenes para entrenar.")
        return

    label_ids = {nombre: idx for idx, nombre in enumerate(set(nombres))}
    labels = [label_ids[nombre] for nombre in nombres]

    recognizer.train(rostros, np.array(labels))
    recognizer.save('modelo_LBPHV2.yml')

    with open('label_ids.txt', 'w') as f:
        for nombre, idx in label_ids.items():
            f.write(f"{nombre}:{idx}\n")

    print("Modelo entrenado y guardado como 'modelo_LBPHV2.yml'")

def Img(folder):
    rostros = []
    nombres = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            try:
                nombre_usuario = filename.split("_")[1]  
                rostros.append(img)
                nombres.append(nombre_usuario)
            except IndexError:
                print(f"No se pudo procesar el archivo {filename}, omitiendo.")
    return rostros, nombres

if __name__ == "__main__":
    train_recognizer()
