FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

RUN mkdir workspace

EXPOSE 8001

ENV PORT=8001
ENV HOST=0.0.0.0

CMD ["python", "server.py"]
