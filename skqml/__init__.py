"""scikit-qml (import skqml): quantum machine learning written from scratch in pure NumPy."""
from .autograd import Tensor, tensor, concat, stack, einsum, no_grad
from .quantum.circuit import Circuit, Observable, X, Y, Z, simulate
from .quantum.functions import expval, probs, amplitudes, statevector, parameter_shift
from .quantum import templates
from . import nn

__version__ = "0.8.0"
from . import models
from . import tuning
from . import datasets
from . import timeseries
from . import text
from . import embeddings
from . import qnlp
