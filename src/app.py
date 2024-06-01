import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from facenet_pytorch import MTCNN
import numpy as np
import torch
from tkinter import messagebox
import os
from opencv_fixer import AutoFix; AutoFix()
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections


def upload_video():
    """
    Uploads a video file and processes its frames.
    
    This function prompts the user to select a video file and then reads the frames from the selected file.
    The frames are stored in a list and then passed to the `process_frames` function for further processing.
    """
    status_bar.config(text="Uploading video...")
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        process_frames(frames)

def upload_frames():
    """
    Uploads frames from a selected directory.

    This function prompts the user to select a directory and then uploads all the image files
    with extensions '.jpg', '.png', or '.jpeg' from that directory. It prints the selected
    directory, the number of frames to be uploaded, and the number of frames uploaded.

    Args:
        None

    Returns:
        None
    """
    status_bar.config(text="Uploading frames... Make sure to double click!")
    dir_path = filedialog.askdirectory(title="Please navigate to and double-click the directory you want to select, and then click 'OK'")
    print(f"Selected directory: {dir_path}")  # Print the selected directory
    files = sorted([os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(('.jpg', '.png', '.jpeg'))]) #check file extn here
    frames = [np.array(Image.open(files[i])) for i in range(len(files))]
    process_frames(frames)

def process_frames(frames):
    """
    Process each frame in the given list of frames and detect faces using MTCNN.

    Args:
        frames (list): A list of frames to be processed.

    Returns:
        None
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"CUDA available: {device == 'cuda'}")  # Print CUDA availability

    # download model
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    # load model
    model = YOLO(model_path)

    # mtcnn = MTCNN(keep_all=True, device=device)
    
    face_locations = []
    
    for frame_count, frame in enumerate(frames):
        # # Detect faces using MTCNN
        # boxes, _ = mtcnn.detect(frame)

        results = model(frame)[0]
        detections = Detections.from_ultralytics(results)
        boxes = detections.xyxy
        if boxes is not None:
            face_locations.append(boxes)
        else:
            face_locations.append([])
        
        status_bar.config(text=f"Processed {frame_count + 1} out of {len(frames)} frames")
        root.update_idletasks()
    
    select_faces_to_blur(frames, face_locations)

def select_faces_to_blur(frames, face_locations, scale_factor=0.5):
    """
    Selects faces to blur in a series of frames.

    Args:
        frames (list): A list of frames (images) to process.
        face_locations (list): A list of face locations for each frame.
        scale_factor (float, optional): The scale factor to resize the frames. Defaults to 0.5.

    Returns:
        None
    """

    selected_faces = []

    # Create a copy of the frames to draw on and scale them down
    frames_copy = [cv2.resize(np.copy(frame), None, fx=scale_factor, fy=scale_factor) for frame in frames]

    # Scale down the face locations
    face_locations_scaled = [[box * scale_factor for box in faces] for faces in face_locations]

    # Initialize the selected status for each face
    face_statuses = [[True for _ in faces] for faces in face_locations_scaled]

    # Create a new Tkinter window
    window = tk.Toplevel(root)
    window.minsize(800, 300)  # Set minimum width to 800 and minimum height to 300

    # Create a label to display the image
    image_label = tk.Label(window)
    image_label.pack()

    status_bar_fs = tk.Label(window, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar_fs.pack(side=tk.BOTTOM, fill=tk.X)
    status_bar_fs.config(text="GREEN boxes ➡️ selected for blurring | RED boxes ➡️ unselected faces. Click on a box to toggle its status.")

    def toggle_face_status(event):
        # Get the frame and face statuses for this image
        frame, faces, statuses, original_faces = image_label.image_info

        # Check each face to see if it was clicked
        for i, box in enumerate(faces):
            x1, y1, x2, y2 = box.astype(int)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                # Toggle the status of this face
                statuses[i] = not statuses[i]

                # Redraw the bounding box for this face
                color = (0, 255, 0) if statuses[i] else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Update the image in the label
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)
                image_label.config(image=photo)
                image_label.image = photo

    # Bind the mouse click event to the toggle_face_status function
    image_label.bind('<Button-1>', toggle_face_status)

    def next_frame(event=None):
        nonlocal current_frame
        current_frame += 1
        if current_frame < len(frames_copy):
            update_frame(current_frame)
        else:
            # If it's the last frame, show a message box
            msg = "You have reached the end of the frames. To go back to the selection process, press Yes. To conclude selection, press No?"
            response = messagebox.askyesno("End of frames", msg)
            if response:  # If user wants to go back to the selection process
                current_frame -= 1
                update_frame(current_frame)
            else:  # If user wants to conclude selection
                window.destroy()  # Close the window

    def prev_frame(event=None):
        nonlocal current_frame
        current_frame -= 1
        if current_frame >= 0:
            update_frame(current_frame)

    def update_frame(i):
        frame, faces, statuses = frames_copy[i], face_locations_scaled[i], face_statuses[i]

        # Draw a bounding box for each face
        for box, status in zip(faces, statuses):
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0) if status else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Convert the frame to a PIL image and display it in the label
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        # Store the frame, faces, statuses, and original faces in the label for the toggle_face_status function
        image_label.image_info = (frame, faces, statuses, face_locations[i])

        # Update the window title with the frame number
        if i == 0:
            window.title(f'Select Faces - Frame {i + 1} / {len(frames_copy)} (Press ➡️ to go to the next frame)')
        else:
            window.title(f'Select Faces - Frame {i + 1} / {len(frames_copy)} (Press ⬅️ for previous, ➡️ for next)')

    # Bind the arrow keys to the next_frame and prev_frame functions
    window.bind('<Right>', next_frame)
    window.bind('<Left>', prev_frame)

    # Start with the first frame
    current_frame = 0
    update_frame(current_frame)

    # Wait for the user to close the window
    window.wait_window()

    # Store the selected faces and use the original bounding boxes
    for i, (faces, statuses, original_faces) in enumerate(zip(face_locations_scaled, face_statuses, face_locations)):
        for j, (box, status, original_box) in enumerate(zip(faces, statuses, original_faces)):
            if status:
                x1, y1, x2, y2 = original_box.astype(int)
                selected_faces.append((i, x1, y1, x2, y2))

    blur_selected_faces(frames, selected_faces)

def blur_selected_faces(frames, selected_faces):
    """
    Blurs the selected faces in the original video frames.

    Args:
        frames (list): List of video frames.
        selected_faces (list): List of tuples containing face coordinates (frame_idx, x1, y1, x2, y2).

    Returns:
        list: List of frames with selected faces blurred.
    """
    print("Started blurring the selected faces in the original video")
    blurred_frames = []
    
    for i, frame in enumerate(frames):
        for (frame_idx, x1, y1, x2, y2) in selected_faces:
            if i == frame_idx:
                face_region = frame[y1:y2, x1:x2]
                if face_region.size > 0:  # Check if face_region is not empty
                    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                    frame[y1:y2, x1:x2] = blurred_face
                else:
                    print(f"Face region is empty for frame {i} and box {x1, y1, x2, y2}")
        
        blurred_frames.append(frame)
    
    display_video(blurred_frames)  # Call display_video function here

def display_video(frames):
    """
    Display a video by showing each frame in a separate window.

    Args:
        frames (list): A list of frames to be displayed.

    Returns:
        None
    """
    video_window = tk.Toplevel(root)
    video_window.title("Blurred Video")
    
    video_label = tk.Label(video_window)
    video_label.pack()
    
    def update_frame(frame_idx):
        if frame_idx < len(frames):
            frame = frames[frame_idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 480))  # Resize frame to 640x480
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            
            video_label.config(image=frame)
            video_label.image = frame
            
            frame_idx += 1
            video_label.after(30, update_frame, frame_idx)
    
    update_frame(0)
    
    save_button = tk.Button(video_window, text="Save Video", command=lambda: save_video(frames))
    save_button.pack()

def save_video(frames):
    """
    Saves a list of frames as a video file.

    Args:
        frames (list): A list of frames to be saved as a video.

    Returns:
        None
    """
    file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Video Files", "*.mp4")])
    if file_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(file_path, fourcc, 30.0, (frames[0].shape[1], frames[0].shape[0]))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print("Video saved successfully!")

root = tk.Tk()
root.title("Face Blur App")
root.geometry("800x600")

upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.pack(pady=20)

# Add the new button to the Tkinter window
upload_frames_button = tk.Button(root, text="Upload Frames", command=upload_frames)
upload_frames_button.pack(pady=20)

status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

root.mainloop()