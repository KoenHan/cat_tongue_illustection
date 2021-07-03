from flask import Flask
from pathlib import Path
import yaml


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = Path.cwd() / "uploads"

app.secret_key = 'cat_tongue'
app.jinja_env.globals.update(zip=zip, enumerate=enumerate)

CONFIG_FILEPATHS = map(lambda filename: Path.cwd() / "config" / filename,
                       ["object_detection.yaml", "image_composition.yaml"])
configs = []
for path in CONFIG_FILEPATHS:
    with open(path) as f:
        config_str = f.read()
    configs.append(yaml.load(config_str, Loader=yaml.SafeLoader))
OBJ_DETECTION_CONFIG, IMG_COMPOSITION_CONFIG = configs


app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024



import sys
sys.path.append(str(Path.cwd()))

import webapp.views
