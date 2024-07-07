from dataclasses import dataclass
from dataset.data_loader import DataLoader
from models.base_model import BaseGameRecommendationModel


@dataclass
class ModelWrapper:
    definition: BaseGameRecommendationModel
    model_save_file_name: str
    data_loader_save_file_name: str
    model: None


def load_and_get_data_loader(app):
    if app.default_data_loader is None:
        app.default_data_loader = DataLoader(get_external_database=True)
    return app.default_data_loader


def load_and_get_model_wrapper(model_wrapper):
    if model_wrapper.model is None:
        model_wrapper.model = model_wrapper.definition()
        model_wrapper.model.load(
            model_wrapper.model_save_file_name, load_published_model=True
        )
        model_data_loader = DataLoader.load_from_file(model_wrapper.data_loader_save_file_name, use_published_models_path=True, load_live_data_loader=True)
        model_wrapper.model.set_data_loader(model_data_loader)
    return model_wrapper
