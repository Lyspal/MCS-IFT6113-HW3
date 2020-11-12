# author: Sylvain Laporte
# program: selectanchors.py
# date: 2020-11-13
# object: Tool for selecting anchor vertex for deformation

import os
import argparse
import json
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
parser.add_argument("--output", "-o", default="anchors.json", help="name of the output file")
parser.add_argument("--debug", default="false", help="run in debug mode")
args = parser.parse_args()

input_file = args.input
output_name = args.output
debug = True if args.debug == "true" else False

# Load the mesh
v, f = load_mesh(input_file)

color = np.array([x for x in range(len(v))])
plot(v, f, c=color, return_plot=True, filename='test.html')

# Create the output object
data = {
	"indices": None,
	"new_positions": None
}

# Choose vertices
print(v)
print(f"Choose vertex indexes from 0 to {len(v) - 1} (separate with spaces):")
choice = str(input())
choice = choice.split(" ")
choice = list(map(int, choice))
data["indices"] = choice

# Choose new position for choosed vertices
new_positions = []

for index in choice:
	print(f"Choose a new position for vertex {index} (original position is: {v[index].tolist()}):")
	print("*** Separate coordinates with spaces.")
	new_pos = str(input())
	new_pos = new_pos.split(" ")
	new_pos = list(map(float, new_pos))
	new_positions.append(new_pos)

# Save new positions
data["new_positions"] = new_positions

with open(output_name, "w+") as f:
	json.dump(data, f)

with open(output_name, "r") as f:
	data = json.load(f)

	print(data)

	for index in data["new_positions"]:
		index = np.array(index)
		print(index)