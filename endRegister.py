from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

@app.route('/registro', methods=['POST'])
def registro():
    data = request.json
    image_data = data['image']
    user_id = data['userId']
    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    count = len([name for name in os.listdir(dataset_folder) if name.startswith(f"user_{user_id}_")]) + 1
    cv2.imwrite(f"{dataset_folder}/user_{user_id}_{count}.jpg", img)

    return jsonify({"mensaje": "Imagen guardada correctamente"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
