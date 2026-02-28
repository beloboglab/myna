import yaml

def load_config(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        config = {}
    return config
