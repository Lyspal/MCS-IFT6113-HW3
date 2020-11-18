# author: Sylvain Laporte
# program: arap_deformation.py
# date: 2020-11-16
# object: Implementation of as-rigid-as-possible deformation

import sys
import argparse
import json
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import inv, svds
import numpy as np
from igl import adjacency_list
from meshplot import offline, plot
from scripts import load_mesh, compute_laplacian, compute_massmatrix, \
    compute_eigenv_sparse

np.set_printoptions(threshold=sys.maxsize)
offline()

def get_unknown_indices(n_vertices, n_unknown, d_b_indices):
    """Computes unknown vertices indices.

    Args:
        n_vertices (int): the number of vertices in the mesh
        n_unknown (int): the number of unknown vertices

    Returns:
        ndarray: an array of unknown indices
    """
    # Get the list of unknown vertex indices
    is_unknown = [True] * n_vertices
    unknown_indices = [None] * n_unknown

    # Set is_unknown for each known vertex
    for index in d_b_indices:
        is_unknown[index] = False

    # Create a list of unknown vertices index
    counter = 0
    for i in range(n_vertices):
        if is_unknown[i]:
            unknown_indices[counter] = i
            counter += 1

    return np.array(unknown_indices)

def compute_arap_displacements(iter, L, d_b_indices, x_bc, v, v_old, f, C, Ri):
    """Computes the displacement d for each vertex.

    Args:
        L (csr_matrix): Laplacian-Beltrami operator
        d_b_indices (ndarray): a list of the known vertices indices
        x_bc (ndarray): the position of the known vertices

    Returns:
        [ndarray]: a vector of the displacements
    """
    # STEP 3: estimate the local rotations Ri after the first iteration
    if iter > 1:
        adjacency = adjacency_list(f)
        Ri = [None] * n_vertices
        for i in range(n_vertices):
            Si = compute_covariance_matrix(v_old, v, i, adjacency[i], C)
            Ui, sigmai, Vti = svds(Si, k=2)
            Ri[i] = Vti.transpose().dot(Ui.transpose())
        Ri = np.array(Ri)

    # STEP 4: solve using Cholesky decomposition (llt)
    sol = np.empty((n_vertices, 3))

    # Set known values in solution
    for i in range(n_known):
        for j in range(3):
            sol[d_b_indices[i], j] = x_bc[i, j]

    # Compute Cholesky decomposition for A
    c, low = cho_factor(L_unknown_unknown)

    # Apply rotation to x_bc
    Ri = Ri[d_b_indices, :, :]
    Rp = np.empty((n_known, 3))
    for i in range(n_known):
        Rp[i] = Ri[i].dot(x_bc[i])

    # Solve for each column of Rp
    for i in range(3):
        x_bc_row = Rp[:, i].reshape(n_known, 1)

        # Compute b in Ax = b for a row in d_bc
        b = L_unknown_known.dot(x_bc_row)

        # Solve Ax = b with Cholesky decomposition
        sol_for_row = cho_solve((c, low), b)

        # Multiply sol by -1
        sol_for_row *= -1

        # Add sol_for_row in solution
        for j in range(sol_for_row.shape[0]):
            sol[unknown_indices[j], i] = sol_for_row[j, 0]
    
    return sol

def compute_guess(n_vertices, n_unknown, L_unknown_unknown, v, unknown_indices):
    """Computes the initial guess based on naive Laplacian editing.

    Args:
        n_vertices (int): the number of vertices in the mesh
        n_unknown (int): the number of unknown displacements
        L_unknown_unknown (csr_matrix): the Laplacian for uknown displacements
        v (ndarray): the mesh vertices
        unknown_indices (ndarray): a list of the unknown vertices indices

    Returns:
        [type]: a n_vertices x 3 matrix of the initial guess
    """
    unknown_vertices = v[unknown_indices, :]

    # Compute both sides of the naive Laplacian editing
    Lp = L_unknown_unknown.dot(unknown_vertices)
    At = L_unknown_unknown.transpose()
    AtA = At.dot(L_unknown_unknown)

    # Solve with solver: Cholesky decomposition (llt)
    guess = np.empty((n_vertices, 3))

    # Compute Cholesky decomposition for AtA
    c, low = cho_factor(AtA)

    # Solve for each row of Lp
    for i in range(3):
        Lp_row = Lp[:, i].reshape(n_unknown, 1)

        # Compute b in Ax = b <=> AtAx = Atb for a row in Lp
        Atb = At.dot(Lp_row)

        # Solve Ax = b with Cholesky decomposition
        sol_for_row = cho_solve((c, low), Atb)

        # Multiply sol by -1
        sol_for_row *= -1

        # Add sol_for_row in solution
        for j in range(sol_for_row.shape[0]):
            idx_test = unknown_indices[j]
            sol_r = sol_for_row[j, 0]
            guess[idx_test, i] = sol_r
    
    return guess

def compute_Ri(n_vertices, v_old, v, C):
    """Computes rotations Ri.

    Args:
        n_vertices (int): number of vertices
        v_old (ndarray): previous vertex positions
        v (ndarray): current vertex positions
        C (csr_matrix): cotangent matrix

    Returns:
        ndarray: an array of rotations
    """
    adjacency = adjacency_list(f)
    Ri = [None] * n_vertices
    for i in range(n_vertices):
        Si = compute_covariance_matrix(v_old, v, i, adjacency[i], C)
        Ui, sigmai, Vti = svds(Si, k=2)
        Ri[i] = Vti.transpose().dot(Ui.transpose())
    return np.array(Ri)

def compute_covariance_matrix(v, guess, current_index, adjacent_vertices, C):
    """Computes the covariance matrix.

    Args:
        v (ndarray): the mesh vertices
        vertex_i (ndarray): the current vertex
        adjacent_vertices (list): a list of the neighbors to vertex_i
        C (csc_matrix): the cotangent matrix

    Returns:
        csc_matrix: the covariance matrix
    """
    diag_weights = C[adjacent_vertices, adjacent_vertices].tolist()

    # Compute D
    Di = diags(diag_weights, [0])

    # Compute Pi
    Pit = np.array([]).reshape(0, 3)
    for j in adjacent_vertices:
        eij = current_index - v[j]
        Pit = np.concatenate((Pit, [eij]))
    Pi = Pit.transpose()

    # Compute Pi_prime
    Pi_prime_t = np.array([]).reshape(0, 3)
    for j in adjacent_vertices:
        eij = guess[current_index] - guess[j]
        Pi_prime_t = np.concatenate((Pi_prime_t, [eij]))

    return csc_matrix(Pi.dot((Di.dot(Pi_prime_t))))

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

# Vertices counts
n_vertices = v.shape[0]
n_known = len(d_b_indices)
n_unknown = n_vertices - n_known

# Get the list of unknown vertex indices
unknown_indices = get_unknown_indices(n_vertices, n_unknown, d_b_indices)

# STEP 1: compute the weights and discrete Laplace-Beltrami operator
C = compute_laplacian(v, f)
M = compute_massmatrix(v, f)

Mi = inv(M)

L = Mi.dot(-C)
L = L.tocsr()

# Extract unknown rows and columns from L
L_intermediate = L[unknown_indices, :]
L_unknown_unknown = L_intermediate.tocsc()[:, unknown_indices].todense()
L_unknown_known = L_intermediate.tocsc()[:, d_b_indices].todense()

# STEP 2: compute the first rotations with initial guess
initial_guess = compute_guess(n_vertices, n_unknown, L_unknown_unknown, v, unknown_indices)
Ri = compute_Ri(n_vertices, initial_guess, v, C)

# Iterate to converge
n_iterations = 3
for i in range(n_iterations):
    iteration = i + 1   # For output file name

    # Conserve previous vertex positions
    v_old = v

    # Compute new vertex positions (STEPS 3 and 4)
    new_p = compute_arap_displacements(i, L, d_b_indices, x_bc, v, v_old, f, C, Ri)
    v = new_p

    # Generate an image of the result
    eigenvalues, eigenvectors = compute_eigenv_sparse(L)
    plot(v, f, c=eigenvectors[:, 4], shading={"wireframe": True},
        return_plot=True, filename=f"{output_file_prefix}-after-arap-it{iteration}.html")