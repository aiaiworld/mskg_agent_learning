FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY benchmarks ./benchmarks
COPY results ./results

EXPOSE 8090

CMD ["python", "src/mskg_agent_learning_demo.py", "--serve", "--host", "0.0.0.0", "--port", "8090"]

