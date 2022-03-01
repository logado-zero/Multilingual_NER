import os
import json
import numpy
import bson
import sys


def ipython_info():
    ip = False
    if 'ipykernel' in sys.modules:
        ip = 'notebook'
    elif 'IPython' in sys.modules:
        ip = 'terminal'
    return ip


def get_tqdm():
    ip = ipython_info()
    if ip == "terminal" or not ip:
        from tqdm import tqdm
        return tqdm
    else:
        try:
            from tqdm import tqdm_notebook
            return tqdm_notebook
        except:
            from tqdm import tqdm
            return tqdm

class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, bson.ObjectId):
            return str(obj)
        else:
            return super(JsonEncoder, self).default(obj)
            
def read_config(config):
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            config = json.load(f)
    return config


def save_config(config, path):
    with open(path, "w") as file:
        json.dump(config, file, cls=JsonEncoder)


def if_none(origin, other):
    return other if origin is None else origin