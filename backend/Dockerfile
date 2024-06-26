FROM python:3.11-slim

ARG FLASK_BACKEND_URL
ARG FLASK_FRONTEND_URL
ARG FLASK_SECRET_KEY
ARG FLASK_STEAM_WEB_API_KEY
ENV FLASK_BACKEND_URL=${FLASK_BACKEND_URL}
ENV FLASK_FRONTEND_URL=${FLASK_FRONTEND_URL}
ENV FLASK_SECRET_KEY=${FLASK_SECRET_KEY}
ENV FLASK_STEAM_WEB_API_KEY=${FLASK_STEAM_WEB_API_KEY}

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 3000

WORKDIR $APP_HOME

RUN apt update && \
    apt -y install build-essential

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

COPY dataset/data_files/users.bin ./dataset/data_files/
COPY dataset/data_files/users_games.bin ./dataset/data_files/
COPY dataset/data_files/games.json ./dataset/data_files/

COPY dataset/__init__.py ./dataset/
COPY dataset/data_loader.py ./dataset/
COPY dataset/scrape/__init__.py ./dataset/scrape/
COPY dataset/scrape/constants.py ./dataset/scrape
COPY dataset/scrape/convert_to_database.py ./dataset/scrape
COPY dataset/scrape/file_manager.py ./dataset/scrape
COPY dataset/scrape/get_data.py ./dataset/scrape
COPY dataset/scrape/merge_all.py ./dataset/scrape
COPY dataset/scrape/merge_data.py ./dataset/scrape
COPY dataset/scrape/serialization.py ./dataset/scrape

COPY utils/__init__.py ./utils/
COPY utils/utils.py ./utils/
COPY utils/firestore.py ./utils/

COPY backend/ ./

CMD exec gunicorn --bind :$PORT --workers 2 --threads 1 --timeout 0 main:app