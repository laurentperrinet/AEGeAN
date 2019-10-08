__version__ = '20190918'
import matplotlib
matplotlib.use('Agg')

from .init import *
from .models import *
from .plot import *
from .utils import *
from .aegean import *
from .ms_ssim import *
import matplotlib.pyplot as plt
# import sys
# sys.path.append("../AEGEAN/")

__all__ = ["utils", "plot", "models", "init", "aegean", "ms_ssim"]


# from .SimpsonsDataset import SimpsonsDataset, FastSimpsonsDataset
