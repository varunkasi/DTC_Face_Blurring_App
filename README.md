# Face Detection and Blurring Project

This project uses the YOLO model for face detection and blurring in videos and images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker with GPU support

### Installation

1. Clone the repository:

gh repo clone varunkasi/DTC_Face_Blurring_App

2. Build the Docker image:

docker build --no-cache -t dtcfaceblurapp:v2 .

3. Run the Docker container:

In a devcontainer inside VSCode - 

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /Local/Code/Directory/:/app/ -v /Local/Video/Directory/:/app/videos --name video_face_blurring_tool_ultralytics  dtcfaceblurapp:v2

In a regular terminal - 

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /Local/Code/Directory/:/app/ -v /Local/Video/Directory/:/app/videos -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --name video_face_blurring_tool_ultralytics  dtcfaceblurapp:v2 python3 /app/src/app.py


### Usage
You can run the application by executing the app.py script in the src directory. The application provides a GUI for uploading videos or image frames, which are then processed for face detection and blurring.


### License
This project is licensed under the MIT License - see the LICENSE.md file for details.

