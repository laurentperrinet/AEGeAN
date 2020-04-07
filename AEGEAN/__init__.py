__version__ = '20191014'
import matplotlib
matplotlib.use('Agg')

from .init import *
from .models import *
from .utils import *
from .aegean import *
import matplotlib.pyplot as plt

__all__ = ["utils", "models", "init", "aegean"]
