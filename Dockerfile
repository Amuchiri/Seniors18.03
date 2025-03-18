# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app


# Copy only the necessary files first (to improve caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip  # ✅ Ensures latest pip version
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]




