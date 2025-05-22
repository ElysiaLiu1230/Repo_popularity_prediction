g Use Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy app files
COPY . /app 

# Install dependencies
RUN pip install --no-cache-dir -r require.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

