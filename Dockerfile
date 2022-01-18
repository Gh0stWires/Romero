FROM tensorflow/tensorflow:2.7.0-gpu

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=0

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./

ENTRYPOINT ["python", "doomgan.py"]