FROM pytorch/pytorch:latest

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
