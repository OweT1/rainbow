# Declare a build argument to hold the Python version
ARG PYTHON_VERSION

# Use the build argument in the FROM instruction
FROM python:${PYTHON_VERSION}-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code from the host to the container at /app
COPY src/edge/ .

# Command to run the application
CMD ["python", "face_detector.py"]