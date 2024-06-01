"""
This module handles the image carousel functionality.
"""

import tkinter as tk
from PIL import Image, ImageTk
from collections import defaultdict


class ImageCarousel:
    def __init__(self, parent):
        self.parent = parent
        self.frames = []
        self.current_frame_index = 0
        self.selected_faces = defaultdict(lambda: [])

    def add_frame(self, frame, faces):
        self.frames.append((frame, faces))
        self.selected_faces[len(self.frames) - 1] = [False] * len(faces)

    def display_frame(self, index, canvas):
        if 0 <= index < len(self.frames):
            self.current_frame_index = index
            frame, faces = self.frames[index]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            for i, (x1, y1, x2, y2) in enumerate(faces):
                color = "green" if self.selected_faces[index][i] else "red"
                canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)

    def toggle_face_selection(self, x, y):
        frame_index = self.current_frame_index
        faces = self.frames[frame_index][1]
        for i, (x1, y1, x2, y2) in enumerate(faces):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_faces[frame_index][i] = not self.selected_faces[frame_index][i]
                break

    def select_all_faces(self):
        for frame_index in self.selected_faces:
            self.selected_faces[frame_index] = [True] * len(self.selected_faces[frame_index])

    def unselect_all_faces(self):
        for frame_index in self.selected_faces:
            self.selected_faces[frame_index] = [False] * len(self.selected_faces[frame_index])

    def get_selected_faces(self):
        return self.selected_faces