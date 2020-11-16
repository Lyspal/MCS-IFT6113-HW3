# author: Sylvain Laporte
# program: selectanchors.py
# date: 2020-11-16
# object: Implementation of biharmonic deformation

import argparse
import json
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import inv
import numpy as np
from meshplot import offline, plot
from scripts import load_mesh, compute_laplacian, compute_massmatrix, \
    compute_eigenv_sparse

offline()

def compute_d_bc(v, x_b_indices, x_b, x_bc):
    """Computes the user displacements.

    Args:
        v (ndarray): mesh vertices
        d_user (ndarray): user selected anchors
        new_positions (ndarray): user selected new positions for anchors

    Returns:
        [ndarray]: a matrix of the known displacements
    """
    displacements = np.zeros((len(x_b_indices), 3))
    for index in range(0, len(x_b)):
        displacements[index] = x_bc[index] - v[x_b_indices[index]]
    return displacements

def minimize_quadratic_energy(K, d_b_indices, d_bc):
    """Computes the displacement d for each vertex.

    Args:
        K (csr_matrix): bilaplacian matrix
        d_b_indices (ndarray): a list of the known vertices indices
        d_bc (ndarray): the displacement of the known vertices

    Returns:
        [ndarray]: a vector of the displacements
    """
    n_vertices = K.shape[0]
    n_known = len(d_b_indices)
    n_unknown = n_vertices - n_known

    # Get the list of unknown vertex indices
    is_unknown = [True] * n_vertices
    unknown_indices = [None] * n_unknown

    # Set is unknown for each known vertex
    for index in d_b_indices:
        is_unknown[index] = False

    # Create a list of unknown vertices index
    counter = 0
    for i in range(n_vertices):
        if is_unknown[i]:
            unknown_indices[counter] = i
            counter += 1

    unknown_indices = np.array(unknown_indices)

    # Extract unknown rows and columns from K to compute A in Ax = b
    A_intermediate = K[unknown_indices, :]
    A_unknown_unknown = A_intermediate.tocsc()[:, unknown_indices].todense()
    A_unknown_known = A_intermediate.tocsc()[:, d_b_indices].todense()

    # Solve using Cholesky decomposition

    sol = np.empty((n_vertices, 3))

    # Set known values in solution
    for i in range(n_known):
        for j in range(3):
            sol[d_b_indices[i], j] = d_bc[i, j]

    # Solve with solver: Cholesky decomposition (llt)
    # Compute Cholesky decomposition for A
    c, low = cho_factor(A_unknown_unknown)

    # Solve for each row of d_bc
    for i in range(3):
        d_bc_row = d_bc[:, i].reshape(n_known, 1)

        # Compute b in Ax = b for a row in d_bc
        b = A_unknown_known * d_bc_row

        # Solve Ax = b with Cholesky decomposition
        sol_for_row = cho_solve((c, low), b)

        # Multiply sol by -1
        sol_for_row *= -1

        # Add sol_for_row in solution
        for j in range(sol_for_row.shape[0]):
            sol[unknown_indices[j], i] = sol_for_row[j, 0]
    
    return sol

# Define command line arguments
parser = argparse.ArgumentParser(description="Run biharmonic deformation")
parser.add_argument("--input", "-i", default="input/bar2.off",
	help="path to input mesh")
parser.add_argument("--anchors", "-a", default="bar2-anchors.json",
    help="path to the anchors' file")
parser.add_argument("--debug", default="false", help="run in debug mode")
args = parser.parse_args()

input_file = args.input
input_file_name = input_file.split(".")[0].split("/")[-1]
anchors_file = args.anchors
output_file_prefix = f"output/{input_file_name}"
debug = True if args.debug == "true" else False

# Load the mesh
v, f = load_mesh(input_file)

# Load the anchors file
anchors_data = None
with open(anchors_file, "r") as file:
    anchors_data = json.load(file)

# Constraints
d_b_indices = anchors_data["indices"]           # indices of the anchors vertices
x_b = np.array([v[i] for i in d_b_indices])     # original positions of the anchors
x_bc = np.array(anchors_data["new_positions"])  # new positions for the anchors

d_bc = compute_d_bc(v, d_b_indices, x_b, x_bc)  # displacement of the anchors

# Compute the bilaplacian
L = compute_laplacian(v, f)
M = compute_massmatrix(v, f)

Mi = inv(M)

K = -L * Mi * -L
K = K.tocsr()

# Generate a preview of the mesh
eigenvalues, eigenvectors = compute_eigenv_sparse(L)
plot(v, f, c=eigenvectors[:, 4], shading={"wireframe": True},
    return_plot=True, filename=f"{output_file_prefix}-before.html")

# Compute the displacements
d = minimize_quadratic_energy(K, d_b_indices, d_bc)

# Compute new vertex positions
v = v + d

# Generate an image of the result
plot(v, f, c=eigenvectors[:, 4], shading={"wireframe": True},
    return_plot=True, filename=f"{output_file_prefix}-after.html")