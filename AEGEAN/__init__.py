__version__ = '20190725'

__all__ = ["utils", "plot", "models", "init", "aegean", "SimpsonsDataset"]

from .SimpsonsDataset import SimpsonsDataset, FastSimpsonsDataset
from .utils import *
from .plot import *
from .models import *
from .init import *
from .aegean import *
