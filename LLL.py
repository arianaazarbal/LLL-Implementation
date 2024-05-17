#HELPER FUNCTIONS
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time

#params
#-vi: non-reduced lattice vector
#-vj_star: graham schmidt reduced lattice vector
#returns: the quantity u_ij
def get_u(vi, vj_star):
    dot_ij = dot(vi, vj_star)
    mag_j = abs(dot(vj_star, vj_star))
    u_ij = dot_ij / mag_j
    return u_ij

#returns: the dot product of two vectors: in this case, a sum over elementwise multiplication operations
def dot(v_1, v_2):
    return np.sum(v_1 * v_2)

#returns: the absolute value of the dot product of a vector with itself
def normSquared(v):
    return abs(dot(v, v))

#returns: the magnitude of the vector
def norm(v):
    return np.sqrt(dot(v, v))

#params
# -index: the index i (into basis) used in the equation v_i* = v_i + sum over j of (u_ij)v_j*
# -basis: the basis that is not graham schmidt reduced
# -gs: the basis where indices 0 -> i - 1 are already graham schmidt reduced
#computes the graham schmidt reduction of a single vector which will be inserted later at index i into the matrix gs.
#returns the updated graham schmidt basis.
def grahamSchmidtVector(index, basis, gs):
    out = basis[index].astype('float64')
    for j in range(index):
        u_ij = get_u(basis[index], gs[j])
        out -= gs[j] * u_ij
    gs[index] = out
    return gs

#params
# -basis: "bad" basis to be reduced using graham schmidt algorithm
#returns the graham scmidt basis (v_1*, ..., v_n*)
def grahamSchmidt(basis):
    gs = np.empty(shape=basis.shape, dtype='float64')
    gs[0] = basis[0]
    n = basis.shape[0]
    for i in range(n):
        gs = grahamSchmidtVector(i, basis, gs)
    return gs

#################################################################################
#LLLNaive
###############################################################################

#input: "bad" basis
#returns: "good" basis of the lattice
#most naive implementation possible. 
def LLLNaive(user_basis):
    #copies the basis so a user's basis is not directly modified (they can access the original basis after calling LLL) 
    basis = np.copy(user_basis)
    gs_basis = grahamSchmidt(basis)
    k = 1
    n = basis.shape[0]

    #iterate through every index (maybe multiple times for some indices) in the basis list
    while k <= n - 1: 
        for j in range(k - 1, -1, -1):
            u_kj = get_u(basis[k], gs_basis[j])
            #computes a reduced vector by rounding the typical graham schmidt factor u_kj so that it is an integer,
            #ensuring the resulting vector remains in the lattice (linear combo of basis elements). 
            basis[k] = basis[k] - np.round(u_kj) * basis[j]

        #not-optimal: re-computes the entire graham scmidt basis when the kth basis vector is being altered.
        #also computes graham schmidt basis vectors for later indices which are useless, as they'll later be overwritten 
        #by an updated computation which takes into account the reduced/updated lattice vectors to be filled in before then. 
        gs_basis = grahamSchmidt(basis)
        u_k = get_u(basis[k], gs_basis[k - 1])
        
        #checks lovasz condition
        left_hand_side = dot(gs_basis[k], gs_basis[k])
        right_hand_side = ((3 / 4) - np.square(u_k)) * dot(gs_basis[k - 1], gs_basis[k - 1])
        lovasz_condition = (left_hand_side >= right_hand_side)
        
        if lovasz_condition:
            #basis is looking "good" so far. move on to reduce the next vector in the lattice. 
            k += 1
        else:
            #basis is not in the correct order. (vector at index k is relatively too large to be at its current position)
            #we must swap v_k with the previous vector v_(k - 1) and re-check if this satisfies LLL conditions. 
            placeholder = np.copy(basis[k])
            basis[k] = basis[k - 1]
            basis[k - 1] = placeholder
            k = max(k - 1, 1)
            gs_basis = grahamSchmidt(basis)
    return basis

###############################################################################
#LLLOptimized
###############################################################################

#input: "bad" basis
#returns: "good" basis of the lattice
#mostly naive, but contains a basic optimization to avoid unnecessary grahamSchmidt calculations. 
def LLLOptimized(user_basis):
    #copies the basis so a user's basis is not directly modified (they can access the original basis after calling LLL) 
    basis = np.copy(user_basis)
    #refrains from computing entire graham schmidt basis prematurely
    gs_basis = np.empty(shape=basis.shape, dtype='float64')
    gs_basis[0] = basis[0]
    k = 1
    n = basis.shape[0]

    #iterate through every index (maybe multiple times for some indices) in the basis list
    while k <= n - 1:
        for j in range(k - 1, -1, -1):
            u_kj = get_u(basis[k], gs_basis[j])
            #computes a reduced vector by rounding the typical graham schmidt factor u_kj so that it is an integer,
            #ensuring the resulting vector remains in the lattice (linear combo of basis elements). 
            basis[k] = basis[k] - np.round(u_kj) * basis[j]

        #optimization. utilizes a 'vectorized' graham schmidt algorithm which allows us to compute
        #and insert a single graham schmidt vector at index k with respect to the already reduced basis vectors 
        #preceding it. 
        gs_basis = grahamSchmidtVector(k, basis, gs_basis)

        u_k = get_u(basis[k], gs_basis[k - 1])
        
        #checks lovasz condition     
        left_hand_side = dot(gs_basis[k], gs_basis[k])
        right_hand_side = ((3 / 4) - np.square(u_k)) * dot(gs_basis[k - 1], gs_basis[k - 1])
        lovasz_condition = (left_hand_side >= right_hand_side)
        
        if lovasz_condition:
            #basis is looking "good" so far. move on to reduce the next vector in the lattice. 
            k += 1
        else:
            #basis is not in the correct order. (vector at index k is relatively too large to be at its current position)
            #we must swap v_k with the previous vector v_(k - 1) and re-check if this satisfies LLL conditions. 
            placeholder = np.copy(basis[k])
            basis[k] = basis[k - 1]
            basis[k-1] = placeholder

            #optimization. only computes graham schmidt reduction for k and k - 1 indices, taking the previous
            #gs_basis as already reduced. 
            gs_basis = grahamSchmidtVector(k, basis, grahamSchmidtVector(k - 1, basis, gs_basis))
            
            k = max(k - 1, 1)
    return basis
