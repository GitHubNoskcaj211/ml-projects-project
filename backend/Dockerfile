FROM python:3.11-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 3000

WORKDIR $APP_HOME

COPY backend/ ./

COPY dataset/data_files/* ./dataset/data_files/

COPY dataset/data_loader.py ./dataset/

COPY utils/utils.py ./utils/

RUN pip install --no-cache-dir -r ./requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app