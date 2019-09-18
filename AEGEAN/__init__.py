__version__ = '20190725'

import sys
sys.path.append("../AEGEAN/")

__all__ = ["utils", "plot", "models", "init", "aegean", "SimpsonsDataset"]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .aegean import *
from .SimpsonsDataset import SimpsonsDataset, FastSimpsonsDataset
from .utils import *
from .plot import *
from .models import *
from .init import *
