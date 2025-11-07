# Import fixtures from model_car conftest to avoid duplication
pytest_plugins = [
    "tests.model_serving.model_server.model_car.conftest",
]
