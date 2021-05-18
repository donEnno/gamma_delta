import time
from joblib import Parallel, delayed, dump

import numpy as np
import pandas as pd
import seaborn as sns
import datashader
import bokeh
import holoviews
import colorcet
import umap
import umap.plot
import matplotlib.pyplot as plt

data_dir = r'C:\Users\Enno\PycharmProjects\gamma_delta\patients'

