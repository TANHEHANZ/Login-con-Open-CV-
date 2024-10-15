import os
import cv2
import numpy as np

dataset_folder = "dataset"
labels_file = "label_ids.txt"x
model_file = "modelo_LBPHV2.yml"

def update_model():
    image_paths = []
    label_ids = {}
    current_id = 0
    x_train = []
    y_labels = []
    for file in os.listdir(dataset_folder):
        if file.endswith("jpg"):
            image_path = os.path.join(dataset_folder, file)
            parts = file.split("_")
            if len(parts) < 2:
                continue
            
            label = parts[1]
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            x_train.append(image)
            y_labels.append(label_ids[label])

    with open(labels_file, 'w') as f:
        for label, id_ in label_ids.items():
            f.write(f"{label}:{id_}\n")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(model_file)

def load_label_ids():
    label_ids = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            for line in f:
                label, id_ = line.strip().split(':')
                label_ids[int(id_)] = label
    return label_ids
