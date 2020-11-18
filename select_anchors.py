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
parser.add_argument("--input", "-i", default="input/bar2.off",
	help="path to input mesh")
parser.add_argument("--debug", default="false", help="run in debug mode")
args = parser.parse_args()

input_file = args.input
input_file_name = input_file.split(".")[0].split("/")[-1]
output_file = f"output/{input_file_name}-anchors.json"
debug = True if args.debug == "true" else False

# Load the mesh
v, f = load_mesh(input_file)

# color = np.array([x for x in range(len(v))])
# plot(v, f, c=color, return_plot=True, filename='test.html')

# Create the output object
data = {
	"indices": [],
	"new_positions": []
}

# Choose vertices
running = True
choice = None

while running:
	print(v)
	print(f"Choose vertex indexes from 0 to {len(v) - 1} (separate with spaces):")
	print("*** Use 'range' in first position to choose a range of vertices.")
	print("*** Eg.: range 1 3 or 1 2 3.")
	choice = str(input())
	choice = choice.split(" ")
	if choice[0] == "range" and len(choice) == 3:
		from_range = int(choice[1])
		to_range = int(choice[2])
		choice = list(range(from_range, to_range + 1))
	else:
		choice = list(map(int, choice))
	data["indices"].extend(choice)

	# Choose new position for choosed vertices
	for index in choice:
		print(f"By how much you want to move vertex {index} (original position is: {v[index].tolist()}):")
		print("*** Separate coordinates with spaces. Use 'same' to keep to vertex in place")
		print("*** Eg.: -5 0 0 to move the vertex 5 units to the left in the x direction.")
		new_pos = str(input())
		if new_pos == "same":
			data["new_positions"].append(v[index].tolist())
		else:
			new_pos = new_pos.split(" ")
			new_pos = v[int(index)] + np.array(list(map(float, new_pos)))
			new_pos = new_pos.tolist()
			data["new_positions"].append(new_pos)

	# Prompt for another vertex selection
	print(f"Want to choose other vertices? (y/n)")
	answer = str(input())
	if answer == "n":
		running = False

with open(output_file, "w+") as f:
	json.dump(data, f)

with open(output_file, "r") as f:
	data = json.load(f)
	for index in data["new_positions"]:
		index = np.array(index)