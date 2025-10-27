import joblib
import shlex
import json
from distutils import util
import logging
from mariadb_kernel.maria_magics.maria_magic import MariaMagic


def _str_to_obj(s):
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        pass
    try:
        return bool(util.strtobool(s))
    except Exception:
        pass
    try:
        return json.loads(s)
    except Exception:
        pass
    if isinstance(s, str) and len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


class LoadModel(MariaMagic):
    """
    %load_model load_path=/tmp/model.joblib [target_key=last_model]

    Loads a locally saved .joblib model into the `data` dictionary.
    """

    def __init__(self, args=""):
        self.args = args
        self.log = logging.getLogger(__name__)

    def type(self):
        return "Line"

    def name(self):
        return "load_model"

    def help(self):
        return "Load a saved model from a local .joblib file."

    def parse_args(self, input_str):
        if not input_str or input_str.strip() == "":
            return {}
        pairs = dict(token.split("=", 1) for token in shlex.split(input_str))
        for k, v in pairs.items():
            pairs[k] = _str_to_obj(v)
        return pairs

    def execute(self, kernel, data):
        try:
            args = self.parse_args(self.args)
        except Exception:
            kernel._send_message("stderr", "Error parsing arguments.")
            return

        load_path = args.get("load_path")
        target_key = args.get("target_key", "last_model")

        if not load_path:
            kernel._send_message("stderr", "You must provide load_path=/path/to/file.joblib")
            return

        try:
            model_obj = joblib.load(load_path)
            data[target_key] = model_obj
            kernel._send_message("stdout", f"Loaded model from {load_path} → data['{target_key}']")
        except Exception as e:
            kernel._send_message("stderr", f"Failed to load model: {e}")
