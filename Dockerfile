# Start from a standard Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# --- THE PERMISSION FIX ---
# Create the .streamlit directory in a writable location and set permissions
RUN mkdir -p /app/.streamlit && \
    chmod -R 777 /app/.streamlit
# -------------------------

# Expose the port that Streamlit runs on
EXPOSE 8501

# The command to run your app
CMD ["streamlit", "run", "app.py"]