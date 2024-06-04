import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
from tkinter import messagebox
import os
import cv2
from ultralytics import YOLO
from supervision import Detections
from tkinter import simpledialog

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
    files = sorted([os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(('.jpg', '.png', '.jpeg'))]) #check file extn here
    frames = [np.array(Image.open(files[i])) for i in range(len(files))]
    process_frames(frames, input_files=sorted([file for file in os.listdir(dir_path) if file.endswith(('.jpg', '.png', '.jpeg'))])) #check file extn here)

def process_frames(frames, input_files=None):
    """
    Process each frame in the given list of frames and detect faces using MTCNN.

    Args:
        frames (list): A list of frames to be processed.
        input_files (list): A list of input file paths.

    Returns:
        None
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"CUDA available: {device == 'cuda'}")  # Print CUDA availability

    # download model
    # model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

    # load model
    model = YOLO('/app/model/model.pt')

    # mtcnn = MTCNN(keep_all=True, device=device)
    
    face_locations = []

    for frame_count, (frame, input_file) in enumerate(zip(frames, input_files)):
        # # Detect faces using MTCNN
        # boxes, _ = mtcnn.detect(frame)

        results = model(frame)[0]
        detections = Detections.from_ultralytics(results)
        boxes = detections.xyxy
        if boxes is not None:
            face_locations.append(list(boxes))
        else:
            face_locations.append([])

        status_bar.config(text=f"Processed {frame_count + 1} out of {len(frames)} frames")
        root.update_idletasks()
    
    select_faces_to_blur(frames, face_locations, input_files)

def select_faces_to_blur(frames, face_locations, input_files=None, scale_factor=0.5):
    """
    Selects faces to blur in a series of frames.

    Args:
        frames (list): A list of frames (images) to process.
        face_locations (list): A list of face locations for each frame.
        scale_factor (float, optional): The scale factor to resize the frames. Defaults to 0.5.

    Returns:
        None
    """
    
    # Ensure that face_locations is a list
    if not isinstance(face_locations, list):
        face_locations = list(face_locations)

    # Add a flag to indicate whether the user is in draw mode
    draw_mode = False
    # Add a variable to store the starting coordinates of the bounding box
    start_coords = None

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
    status_bar_fs.config(text="GREEN boxes -> selected for blurring | RED boxes -> unselected faces. Click on a box to toggle its status.")

    def toggle_face_status(event):
        nonlocal draw_mode, start_coords

        # Get the frame and face statuses for this image
        frame, faces, statuses, original_faces = image_label.image_info

        if draw_mode:
        # If the user is in draw mode, start drawing a bounding box
            start_coords = (event.x, event.y)
        else:
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

    def draw_box(event):
        nonlocal draw_mode, start_coords
        if draw_mode and start_coords is not None:
            frame, faces, statuses, original_faces = image_label.image_info
            # Create a copy of the frame
            frame_copy = frame.copy()
            # Draw the box on the copy of the frame
            x1, y1 = start_coords
            x2, y2 = event.x, event.y
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Update the image in the label
            image = Image.fromarray(frame_copy)
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo

    def end_draw(event):
        nonlocal draw_mode, start_coords
        if draw_mode and start_coords is not None:
            frame, faces, statuses, original_faces = image_label.image_info
            # Draw the final box on the original frame
            x1, y1 = start_coords
            x2, y2 = event.x, event.y
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add the new box to the list of boxes for this frame
            new_box = list(np.array([x1, y1, x2, y2]) / scale_factor)
            face_locations[current_frame].append(new_box)
            # Add the new box to the faces list
            faces.append(np.array([x1, y1, x2, y2]))
            # Add a new status for this box
            face_statuses[current_frame].append(True)
            # Reset the start coordinates
            start_coords = None
            # Update the frame
            update_frame(current_frame)
            # End the draw mode
            toggle_draw_mode()

    def toggle_draw_mode():
        nonlocal draw_mode
        # Toggle the draw mode
        draw_mode = not draw_mode
        # Update the button text based on the draw mode
        draw_button.config(text="End Draw Mode" if draw_mode else "Start Draw Mode")

    # Add a button to toggle the draw mode
    draw_button = tk.Button(window, text="Start Draw Mode", command=toggle_draw_mode)
    draw_button.pack()

    # Bind the mouse events to the appropriate functions
    image_label.bind('<B1-Motion>', draw_box)
    image_label.bind('<ButtonRelease-1>', end_draw)

    def next_frame(event=None):
        nonlocal current_frame
        # Check if the draw mode is active
        if draw_mode:
            # If it is, prompt the user to end the draw mode before proceeding
            messagebox.showinfo("End Draw Mode", "Please end the draw mode before proceeding to the next frame.")
        else:
            # If it's not, proceed to the next frame
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
        # Check if the draw mode is active
        if draw_mode:
            # If it is, prompt the user to end the draw mode before proceeding
            messagebox.showinfo("End Draw Mode", "Please end the draw mode before proceeding to the previous frame.")
        else:
            # If it's not, proceed to the previous frame
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
                x1, y1, x2, y2 = np.array(original_box).astype(int)
                selected_faces.append((i, x1, y1, x2, y2))

    blur_selected_faces(frames, selected_faces, input_files)

def blur_selected_faces(frames, selected_faces, input_files=None):
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
    
    display_video(blurred_frames, input_files)  # Call display_video function here

def display_video(frames, input_files=None):
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
    
    save_button = tk.Button(video_window, text="Save Video", command=lambda: save_video(frames, input_files))
    save_button.pack()

def save_video(frames, input_files=None):
    """
    Saves a list of frames as a video file.

    Args:
        frames (list): A list of frames to be saved as a video.

    Returns:
        None
    """
    # After processing all frames
    if input_files is not None:
        # Prompt the user to select a directory to save the frames to
        dir_path = filedialog.askdirectory(title="Please select a directory to save the frames to")
        # Iterate over the frames and input_files simultaneously
        for frame, file_name in zip(frames, input_files):
            print(os.path.join(dir_path, "blurred_", file_name))
            # Save the frame to the selected directory with the file name from input_files
            cv2.imwrite(os.path.join(dir_path, "blurred_" + file_name), frame)
    else:
        # Save the frames as a video
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