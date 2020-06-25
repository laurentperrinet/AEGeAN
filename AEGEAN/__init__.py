__version__ = '20191014'

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .init import *
from .models import *
from .utils import *
from .aegean import *

__all__ = ["utils", "models", "init", "aegean"]
