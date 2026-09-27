FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED=True

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the web service on container startup.
# Use $PORT environment variable to bind to the correct port assigned by Cloud Run.
# Proxy headers keep redirects on https behind Cloud Run's front end.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips="*"
