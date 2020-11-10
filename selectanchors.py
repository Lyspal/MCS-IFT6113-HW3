# author: Sylvain Laporte
# program: selectanchors.py
# date: 2020-11-13
# object: Tool for selecting anchor vertex for deformation

import os
import argparse
import igl
import scipy as sp
import numpy as np
from meshplot import offline, plot, subplot, interact
from scripts import load_mesh

root_folder = os.getcwd()
offline()

# Define command line arguments.
parser = argparse.ArgumentParser(description="Select anchor vertices")
parser.add_argument("--input", "-i", default="input/cube.obj",
	help="path to input mesh")
parser.add_argument("--output", "-o", default="test", help="name of the output file")
parser.add_argument("--debug", default="false", help="run in debug mode")
args = parser.parse_args()

input_file = args.input
output_name = args.output
debug = True if args.debug == "true" else False

v, f = load_mesh(input_file)

color = np.array([x for x in range(len(v))])

print(color)

plot(v, f, c=color, return_plot=True, filename='test.html')