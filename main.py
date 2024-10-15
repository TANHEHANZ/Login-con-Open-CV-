from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import base64
from flask_cors import CORS  # Importa CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para toda la aplicación

# Cargar el reconocedor LBPH previamente entrenado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('modelo_LBPH.yml')

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Aquí se debe asignar correctamente los nombres al model
label_ids = {"Hanz": 0, "OtroUsuario": 1}  # Ejemplo

@app.route('/reconocer', methods=['POST'])
def reconocer():
    # Recibe una imagen en formato base64
    image_data = request.json['image']
    
    # Convertir imagen de base64 a numpy array
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Detectar rostros en la imagen recibida
    face_locations = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(face_locations) == 0:
        return jsonify({"error": "No se detectó ningún rostro"})

    for (x, y, w, h) in face_locations:
        rostro_recortado = img[y:y+h, x:x+w]
        label_predicho, confianza = recognizer.predict(rostro_recortado)

        if confianza < 100:  # Usuario reconocido
            nombre_predicho = [nombre for nombre, label in label_ids.items() if label == label_predicho][0]
            return jsonify({"nombre": nombre_predicho, "mensaje": "Usuario reconocido"})
        else:
            return jsonify({"nombre": "Desconocido", "mensaje": "Vete de mi app"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
