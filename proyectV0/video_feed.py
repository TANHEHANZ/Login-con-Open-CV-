import cv2
from PIL import Image, ImageTk
import tkinter as tk

class VideoFeed:
    def __init__(self, video_source, root):
        self.vid = cv2.VideoCapture(video_source)
        self.root = root
        self.canvas = tk.Canvas(root, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        return self.vid.read()

    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

    def update(self):
        ret, frame = self.get_frame()
        if ret:
            self.show_frame(frame)
        self.root.after(10, self.update)

    def release(self):
        if self.vid.isOpened():
            self.vid.release()
