__version__ = '20190725'

import sys
sys.path.append("../AEGEAN/")

__all__ = ["utils", "plot", "models", "init", "aegean", "SimpsonsDataset"]

from .aegean import *
from .SimpsonsDataset import SimpsonsDataset, FastSimpsonsDataset
from .utils import *
from .plot import *
from .models import *
from .init import *

