FROM python:3.11-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 3001

WORKDIR $APP_HOME

RUN apt update && \
    apt -y install build-essential

COPY ml_backend/requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

COPY dataset/data_files/users.bin ./dataset/data_files/
COPY dataset/data_files/users_games.bin ./dataset/data_files/
COPY dataset/data_files/games.json ./dataset/data_files/
COPY models/published_recommendation_models ./models/published_recommendation_models/

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

COPY models/__init__.py ./models/
COPY models/base_model.py ./models/
COPY models/common_neighbors_model.py ./models/
COPY models/ncf_model.py ./models/
COPY models/ncf_singlenode.py ./models/
COPY models/popularity_model.py ./models/
COPY models/random_model.py ./models/

COPY utils/__init__.py ./utils/
COPY utils/utils.py ./utils/
COPY utils/firestore.py ./utils/

COPY ml_backend/ ./

CMD exec gunicorn --bind :$PORT --workers 2 --threads 1 --timeout 0 main:app