FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

EXPOSE 8005

ENV PORT=8005
ENV HOST=0.0.0.0

CMD ["python", "server.py"]
