FROM python:3.11-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 3000

WORKDIR $APP_HOME

COPY backend/ ./

COPY dataset/data_files/* ./dataset/data_files/

COPY dataset/data_loader.py ./dataset/

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