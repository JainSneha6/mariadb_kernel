import joblib
import shlex
import json
import time
from distutils import util
import logging
from mariadb_kernel.maria_magics.maria_magic import MariaMagic
import os

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


class SaveModel(MariaMagic):
    """
    %save_model model_name_in_data=last_model save_path=/tmp/model.joblib [overwrite=True|False]

    Saves a trained model (from the `data` dict) to a local file using joblib.
    """

    def __init__(self, args=""):
        self.args = args
        self.log = logging.getLogger(__name__)

    def type(self):
        return "Line"

    def name(self):
        return "save_model"

    def help(self):
        return "Save a trained model to a local .joblib file."

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

        model_key = args.get("model_name_in_data", "last_model")
        save_path = args.get("save_path")
        overwrite = bool(args.get("overwrite", False))

        

        if not save_path:
            kernel._send_message("stderr", "You must provide save_path=/path/to/file.joblib")
            return

        model_obj = data.get(model_key)
        if model_obj is None:
            kernel._send_message("stderr", f"No model found in data['{model_key}'].")
            return

        # If file exists and overwrite=False
        import os
        if os.path.exists(save_path) and not overwrite:
            kernel._send_message("stderr", f"File {save_path} already exists. Use overwrite=True to replace it.")
            return

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model_obj, save_path)
            kernel._send_message("stdout", f"Model from data['{model_key}'] saved to {save_path}")
        except Exception as e:
            kernel._send_message("stderr", f"Failed to save model: {e}")
