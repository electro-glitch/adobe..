FROM --platform=linux/amd64 python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --no-cache-dir PyMuPDF sentence-transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
ENTRYPOINT ["python", "process_docs.py"]
