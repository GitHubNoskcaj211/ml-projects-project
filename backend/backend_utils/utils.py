from dataclasses import dataclass
from dataset.data_loader import DataLoader
from models.base_model import BaseGameRecommendationModel
import random


@dataclass
class ModelWrapper:
    definition: BaseGameRecommendationModel
    model_save_file_name: str
    data_loader_save_file_name: str
    model: None


def load_and_get_data_loader(app):
    if app.default_data_loader is None:
        app.default_data_loader = DataLoader(app=app, get_external_database=True)
    return app.default_data_loader


def load_and_get_random_model_wrapper(app):
    selected_model_wrapper = random.choice(app.model_wrappers)
    if selected_model_wrapper.model is None:
        selected_model_wrapper.model = selected_model_wrapper.definition()
        selected_model_wrapper.model.load(
            selected_model_wrapper.model_save_file_name, load_published_model=True
        )
        model_data_loader = DataLoader.load_from_file(selected_model_wrapper.data_loader_save_file_name, load_live_data_loader=True)
        model_data_loader.app = app
        selected_model_wrapper.model.set_data_loader(model_data_loader)
    return selected_model_wrapper
