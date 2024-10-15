from flask import Flask, render_template, request, redirect, url_for
import cv2
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        
        capture_faces(user_id)
        
        return redirect(url_for('index'))
    return render_template('register.html')

dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
def capture_faces(user_id):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara o no hay permisos .")
    exit()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    user_id = input('Ingresa tu ID de usuario: ')

    count = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturando la imagen.")
            break

        resized_frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            count += 1

            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{dataset_folder}/user_{user_id}_{count}.jpg", face_img)

        cv2.imshow('Rostro', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(debug=True)