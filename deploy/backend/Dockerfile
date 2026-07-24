FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py .
COPY rubric.json .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
