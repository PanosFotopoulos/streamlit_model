FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py utils.py label_decoder.json model_quantized.onnx requirements.txt ./
COPY test/ ./test

EXPOSE 8501

RUN pip install --no-cache-dir -r requirements.txt

CMD streamlit run app.py