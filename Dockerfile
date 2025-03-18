# Use a Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PortAudio and PyAudio
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files
COPY . .

# Expose port (if needed)
EXPOSE 8080

# Run the application
CMD ["python", "app.py"]





