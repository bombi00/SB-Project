import json
from sklearn.ensemble import HistGradientBoostingClassifier

def load_config(config_path="./data/script/configuration.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def build_model(config_path="./data/script/configuration.json"):
    config = load_config(config_path)
    return HistGradientBoostingClassifier(
        max_iter=config.get("max_iter"),
        max_depth=config.get("max_depth"),
        learning_rate=config.get("learning_rate_hgbc"),
    )