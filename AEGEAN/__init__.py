__version__ = '20190918'

from .init import *
from .models import *
from .plot import *
from .utils import *
from .aegean import *
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("../AEGEAN/")

__all__ = ["utils", "plot", "models", "init", "aegean"]

matplotlib.use('Agg')

# from .SimpsonsDataset import SimpsonsDataset, FastSimpsonsDataset
