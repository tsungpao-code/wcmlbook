import json
from importlib_resources import files
import numpy as np
import torch
import models

SPEED_OF_LIGHT=torch.tensor(299792458)
PI=torch.tensor(3.141592653589793)