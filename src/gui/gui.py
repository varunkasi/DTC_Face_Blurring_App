"""
This module contains the main GUI layout and initialization code.

- create_main_window Function:
    - Main Frame: Split into left and right containers.
    - Left Top Container: Contains the buttons for video upload, playback controls, and face detection, along with a label to display video frames.
    - Video Upload Button: Opens a file dialog to select a video file and loads the video.
    - Playback Control Buttons: Play, pause, stop, rewind, and forward the video.
    - Face Detection Button: Detects and displays faces in the current frame.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
# from gui.video_playback import VideoProcessor
from face_detection.face_detection import FaceDetector
from gui.image_carousel import ImageCarousel
from face_blurring.face_selection import FaceSelection
from face_blurring.face_blurring import FaceBlurring
# from gui.final_video_playback import FinalVideoProcessor
import customtkinter as ctk
from tkVideoPlayer import TkinterVideo
import datetime
import threading

# ToolTip class for displaying tooltips on buttons - example: ToolTip(playback_button, "Play Video")
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="#ffffff", relief="solid", borderwidth=1)
        label.pack(ipadx=1)

    def leave(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


def create_main_window(root):
    def get_frame_rate(video_path):
        video = cv2.VideoCapture(video_path)
        # Get frame rate
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps
    def update_duration(event):
        """ updates the duration after finding the duration """
        duration = vid_player.video_info()["duration"] * root.fps
        # end_time["text"] = str(datetime.timedelta(seconds=duration))
        end_time["text"] = int(duration)
        progress_slider["to"] = duration

    def update_scale(event):
        """ updates the scale value """
        progress_value.set(vid_player.current_frame_number())

    def update_progress_slider():
        while True:
            progress_value.set(vid_player.current_frame_number())
            # time.sleep(0.1)  # Adjust the delay as needed

    def test_frame(event):
        """ updates the scale value """
        pass

    def load_video():
        """ loads the video """
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])

        # print the file path
        print(file_path)

        if file_path:
            root.fps = get_frame_rate(file_path)
            #print the fps
            print(f"FPS: {root.fps}")
            vid_player.load(file_path)
            progress_slider.config(to=0, from_=0)
            play_pause_btn.configure(text="Play")
            progress_value.set(0)

    def seek(value):
        """ used to seek a specific timeframe """
        # print(f"Seek value: {type(value)}")
        vid_player.seek(int(int(value)/root.fps))

    def skip(value: int):
        """ used to skip a specific timeframe """
        vid_player.seek(int(progress_slider.get())+value)
        progress_value.set(progress_slider.getcap.isOpened()() + value)

    def skip_frames(n_frames: int):
        """ used to skip a specific number of frames """
        # frame_time = 1 / root.fps  # time duration of one frame
        vid_player.seek(int((progress_slider.get() + n_frames)/root.fps))
        print(f"Progress_slider.get() value: {int(progress_slider.get())}")
        progress_value.set(progress_slider.get() + n_frames)

    def play_pause():
        """ pauses and plays """
        if vid_player.is_paused():
            vid_player.play()
            play_pause_btn.configure(text="Pause")

        else:
            vid_player.pause()
            play_pause_btn.configure(text="Play")

    def video_ended(event):
        """ handle video ended """
        progress_slider.set(progress_slider["to"])
        play_pause_btn.configure(text = "Play")
        progress_slider.set(0)

    main_frame = ctk.CTkFrame(root)
    main_frame.pack(fill='both', expand=True)

    left_container = ctk.CTkFrame(main_frame)
    left_container.pack(side='left', fill='both', expand=True)

    right_container = ctk.CTkFrame(main_frame)
    right_container.pack(side='right', fill='both', expand=True)

    # Add a vertical separator between left and right containers
    separator = ttk.Separator(main_frame, orient='vertical')
    separator.pack(side='left', fill='y', padx=5)

    # Create a Frame with a fixed height
    fixed_height_frame = tk.Frame(left_container, height=600)

    # Pack left_top_container inside the fixed height frame
    left_top_container = ctk.CTkFrame(fixed_height_frame)
    left_top_container.pack(side='top', fill='both', expand=True)

    # Pack the fixed height frame into left_container
    fixed_height_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents
    fixed_height_frame.pack(side='top', fill='x')

    # Add a horizontal separator between left top and left bottom containers
    separator = ttk.Separator(left_container, orient='horizontal')
    separator.pack(side='top', fill='x', pady=5)

    left_bottom_container = ctk.CTkFrame(left_container)
    left_bottom_container.pack(side='bottom', fill='both', expand=True)

    # Button to load video
    load_button = ctk.CTkButton(left_top_container, text="Load Video", command=load_video)
    load_button.pack(pady=5)

    # Playback controls
    controls_container = ctk.CTkFrame(left_top_container)
    controls_container.pack(fill='x', pady=5)

    # skip_minus_1sec = ctk.CTkButton(controls_container, text="Rewind 1 sec", command=lambda: skip(-1))
    # skip_minus_1sec.pack(side='left', padx=5)

    skip_minus_1frame = ctk.CTkButton(controls_container, text="Rewind 1 frame", command=lambda: skip_frames(-1))
    skip_minus_1frame.pack(side='left', padx=5)

    start_time = ttk.Label(controls_container, text=str(0))
    start_time.pack(side='left', padx=5)

    progress_value = tk.IntVar(controls_container)

    progress_slider = tk.Scale(controls_container, variable=progress_value, from_=0, to=0, orient="horizontal", command=seek)
    progress_slider.pack(side='left', fill="x", expand=True)

    end_time = ttk.Label(controls_container, text=str(0))
    end_time.pack(side='left', padx=5)

    # skip_plus_1sec = ctk.CTkButton(controls_container, text="Forward 1 sec", command=lambda: skip(1))
    # skip_plus_1sec.pack(side='left', padx=5)

    skip_plus_1frame = ctk.CTkButton(controls_container, text="Forward 1 frame", command=lambda: skip_frames(1))
    skip_plus_1frame.pack(side='left', padx=5)

    play_pause_btn = ctk.CTkButton(left_top_container, text="Play", command=play_pause)
    play_pause_btn.pack(pady=5)

    # Create a new CTkFrame inside left_top_container
    # video_frame = ctk.CTkFrame(left_top_container)
    # video_frame.pack(fill="both", expand=True, pady=5)

    # Video player in left_top_container
    vid_player = TkinterVideo(master=left_top_container)
    vid_player.set_size((640, 480))
    vid_player.pack(fill='both', expand=True)

    vid_player.bind("<<Duration>>", update_duration)
    # vid_player.bind("<<SecondChanged>>", update_scale)
    vid_player.bind("<<Ended>>", video_ended )
    # vid_player.bind("<<FrameGenerated>>", update_scale)

    # Start the progress slider update thread
    threading.Thread(target=update_progress_slider, daemon=True).start()



    # def detect_faces():
    #     frame = video_processor.get_frame()
    #     if frame is not None:
    #         frame_with_boxes, boxes = video_processor.face_detector.detect_faces(frame)
    #         if frame_with_boxes is not None:
    #             image_carousel.add_frame(frame, boxes)
    #             image_carousel.display_frame(image_carousel.current_frame_index, canvas)
    #         else:
    #             messagebox.showwarning("Warning", "No faces detected.")
    #     else:
    #         messagebox.showwarning("Warning", "No frame to display or end of video reached.")

    def select_all_faces():
        image_carousel.select_all_faces()
        image_carousel.display_frame(image_carousel.current_frame_index, canvas)

    def unselect_all_faces():
        image_carousel.unselect_all_faces()
        image_carousel.display_frame(image_carousel.current_frame_index, canvas)

    def save_face_selection():
        selected_faces = face_selection.save_selections()
        messagebox.showinfo("Success", f"Face selections saved: {selected_faces}")

    def on_canvas_click(event, image_carousel, canvas):
        image_carousel.toggle_face_selection(event.x, event.y)
        image_carousel.display_frame(image_carousel.current_frame_index, canvas)

    # Create canvas for displaying frames with bounding boxes
    canvas = ctk.CTkCanvas(left_bottom_container)
    canvas.pack(fill='both', expand=True)
    canvas.bind("<Button-1>", lambda event: on_canvas_click(event, image_carousel, canvas))

    # Select/unselect buttons
    select_all_button = ttk.Button(left_bottom_container, text="Select All Faces", command=select_all_faces)
    select_all_button.pack(pady=5)

    unselect_all_button = ttk.Button(left_bottom_container, text="Unselect All Faces", command=unselect_all_faces)
    unselect_all_button.pack(pady=5)

    save_selection_button = ttk.Button(left_bottom_container, text="Save Face Selection", command=save_face_selection)
    save_selection_button.pack(pady=5)

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Video Processing GUI")
    root.geometry("1200x800")
    create_main_window(root)
    root.mainloop()