# Same Python version as the venv used to train the model
FROM python:3.13-slim

WORKDIR /root/code

# Show Python's output in the Docker logs straight away
ENV PYTHONUNBUFFERED=1

# Install the packages first (Docker caches this step between builds)
COPY code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app and the model
COPY code/ .

CMD ["python", "app.py"]
