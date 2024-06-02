# Use the NGC PyTorch image as a base image
FROM ultralytics/ultralytics:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Expose the port for VSCode Remote Development
EXPOSE 22

# Start a shell by default
CMD ["/bin/bash"]
