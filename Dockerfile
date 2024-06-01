# Use the NGC PyTorch image as a base image
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y libgtk2.0-dev pkg-config && \
    apt-get install --reinstall -y libopencv-dev python3-opencv && \
    apt-get install -y ffmpeg libgl1-mesa-glx python3-tk && \
    apt-get clean

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Expose the port for VSCode Remote Development
EXPOSE 22

# Start a shell by default
CMD ["/bin/bash"]
