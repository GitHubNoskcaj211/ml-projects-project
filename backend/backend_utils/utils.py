from dataset.data_loader import DataLoader


def load_and_get_data_loader(app):
    if app.default_data_loader is None:
        app.default_data_loader = DataLoader(get_external_database=True)
    return app.default_data_loader
