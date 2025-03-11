## Introduction

The Lenstra–Lenstra–Lovász (LLL) lattice reduction algorithm plays a key role in the realm of lattice-based cryptography, a branch of post-quantum cryptography. 
The hard problems required to hack many lattice-based cryptosystems, including NTRU, are the Shortest Vector Problem (SVP) and the Closest Vector Problem (CVP). 
These problems are critically "easy" to solve if one has a "good" basis for the lattice, and difficult to solve if one has a "bad" basis for the lattice. 
The LLL reduction algorithm provides a means to find an approximately 'good' basis for a lattice in polynomial time, and thus attack a cryptosystem. This good basis has:  

  a) a small first basis vector (specifically one with a magnitude less than or equal to $2^{\frac{n-1}{2}}||\mathbf{v}||$, where $\mathbf{v}$ is the true smallest vector in the lattice)
  
  b) basis vectors growing in size towards the end of the list
  
  c) approximately orthogonal basis vectors.
  
Thus, LLL's existence is taken into account when formalizing security guarantees for cryptosystems, espeically for the post-quantum computing era.   
## Implementation
My implementation of LLL draws heavily from the Gram-Schmidt Algorithm in Linear Algebra. We begin as though we are implementing Gram-Schmidt, but we critically round the factor to an integer so that the resulting vector remains in the lattice. This means that the resultant is not strictly orthogonal to the prior vectors, like in Gram-Schmidt. 
## Tests
I implemented functions to check these two conditions to test my LLL implementation. I use a testingOracle function (omitted here but in the code) which generates large numbers of random tests and feeds the results into these functions to make sure they are LLL reduced.
