# Start from an official, lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy just the requirements file first (not the whole project yet)
COPY requirements.txt .

# Install all the Python packages this project needs
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project into the container
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Tell Docker which port the Streamlit app will run on
EXPOSE 8501

# The command that runs when the container starts
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]