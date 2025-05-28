# Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy app, models, and requirements into image
COPY app /app
COPY models /models
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
