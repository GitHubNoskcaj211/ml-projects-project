FROM python:3.11-slim

ARG FLASK_BACKEND_URL
ARG FLASK_FRONTEND_URL
ARG FLASK_SECRET_KEY
ENV FLASK_BACKEND_URL=${FLASK_BACKEND_URL}
ENV FLASK_FRONTEND_URL=${FLASK_FRONTEND_URL}
ENV FLASK_SECRET_KEY=${FLASK_SECRET_KEY}

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 3000

WORKDIR $APP_HOME

COPY backend/ ./

COPY dataset/data_files/global_database.db ./dataset/data_files/

COPY dataset/__init__.py ./dataset/
COPY dataset/data_loader.py ./dataset/
COPY dataset/scrape/__init__.py ./dataset/scrape/
COPY dataset/scrape/constants.py ./dataset/scrape
COPY dataset/scrape/convert_to_database.py ./dataset/scrape
COPY dataset/scrape/file_manager.py ./dataset/scrape
COPY dataset/scrape/get_data.py ./dataset/scrape
COPY dataset/scrape/merge_all.py ./dataset/scrape
COPY dataset/scrape/merge_data.py ./dataset/scrape

COPY models/published_recommendation_models ./models/published_recommendation_models/

COPY models/__init__.py ./models/
COPY models/base_model.py ./models/
COPY models/collaborative_filtering_model.py ./models/
COPY models/common_neighbors_model.py ./models/
COPY models/ncf_model.py ./models/
COPY models/ncf_singlenode.py ./models/
COPY models/popularity_model.py ./models/
COPY models/random_model.py ./models/

COPY utils/__init__.py ./utils/
COPY utils/utils.py ./utils/

RUN pip install --no-cache-dir -r ./requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app