from dataclasses import dataclass
from dataset.data_loader import DataLoader
from models.base_model import BaseGameRecommendationModel
import random


@dataclass
class ModelWrapper:
    definition: BaseGameRecommendationModel
    save_file_name: float
    model: None


def load_and_get_data_loader(app):
    if app.data_loader is None:
        app.data_loader = DataLoader(app=app, get_init_database=True)
    return app.data_loader


def load_and_get_random_model_wrapper(app):
    selected_model_wrapper = random.choice(app.model_wrappers)
    if selected_model_wrapper.model is None:
        selected_model_wrapper.model = selected_model_wrapper.definition()
        selected_model_wrapper.model.load(
            selected_model_wrapper.save_file_name, load_published_model=True
        )
        selected_model_wrapper.model.set_data_loader(load_and_get_data_loader(app))
    return selected_model_wrapper
