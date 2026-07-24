import json
from pathlib import Path


def load_rubric():
    return json.loads(Path(__file__).with_name("rubric.json").read_text())
