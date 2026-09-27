# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install system dependencies required by opencv-python and other X11 components
# This comprehensive list covers common requirements for cv2 in slim environments
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libice6 \
    libxrandr2 \
    libxi6 \
    libxcursor1 \
    libxfixes3 \
    libxinerama1 \
    libxcomposite1 \
    libxdamage1 \
    libatk1.0-0 \
    libpango1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit when the container launches
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
