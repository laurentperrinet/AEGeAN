__version__ = '20190918'

import sys
sys.path.append("../AEGEAN/")

__all__ = ["utils", "plot", "models", "init", "aegean"]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .aegean import *
# from .SimpsonsDataset import SimpsonsDataset, FastSimpsonsDataset
from .utils import *
from .plot import *
from .models import *
from .init import *
