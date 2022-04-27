from collections import namedtuple
from flame_utils.FLAME_PyTorch_bonseyes import FLAME
from flame_utils.FLAME_PyTorch_bonseyes.config import get_config
import torch
import json

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
config = get_config()

with open("flame_config.json", 'wt') as f:
    json.dump(vars(config), f, indent=4)

    
# def auto_namedtuple(classname='flame_config', **kwargs):
#     return namedtuple(classname, tuple(kwargs))(**kwargs)

# config_tuple = auto_namedtuple(**vars(config))
# print(type(config_tuple))
# print(config_tuple)