# author: Sylvain Laporte
# program: selectanchors.py
# date: 2020-11-13
# object: Implementation of biharmonic deformation

import os
import argparse
import json
import igl
import scipy as sp
import numpy as np
from meshplot import offline, plot
from scripts import load_mesh

offline()

def compute_hat_d_user(v, anchors, new_anchor_positions):
    """Computes the user displacements.

    Args:
        v (ndarray): mesh vertices
        d_user (ndarray): user selected anchors
        new_positions (ndarray): user selected new positions for anchors

    Returns:
        [type]: [description]
    """
    displacements = np.zeros((len(d_user), 3))
    for index in range(len(anchors)):
        displacements[index] = new_anchor_positions[index] - v[anchors[index]]
    return displacements


# Define command line arguments.
parser = argparse.ArgumentParser(description="Run biharmonic deformation")
parser.add_argument("--input", "-i", default="input/cube.obj",
	help="path to input mesh")
parser.add_argument("--anchors", "-a", default="anchors.json",
    help="path to the anchors' file")
parser.add_argument("--output", "-o", default="test", help="name of the output file")
parser.add_argument("--debug", default="false", help="run in debug mode")
args = parser.parse_args()

input_file = args.input
anchors_file = args.anchors
output_name = args.output
debug = True if args.debug == "true" else False

# Load the mesh
v, f = load_mesh(input_file)

color = np.array([x for x in range(len(v))])
plot(v, f, c=color, return_plot=True, filename='test.html')

# Load the anchors file
anchors_data = None
with open(anchors_file, "r") as f:
    anchors_data = json.load(f)

d = None            # unknown vector of displacement

# Constraints
d_user = np.array(anchors_data["indices"])    # indexes of user fixed vertices
new_positions = np.array(anchors_data["new_positions"])
hat_d_user = compute_hat_d_user(v, d_user, new_positions)   # coordinates specified by user

L = None
M = None




print(d_user)
print(hat_d_user)

