#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from utils.plotting import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("file_path", help="path to file")
args = parser.parse_args()

buf = np.load(args.file_path)
plot_iq(buf)
plt.show()
