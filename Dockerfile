FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predictor.py data_processor.py ./
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "predictor.py"]