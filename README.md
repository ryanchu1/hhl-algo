# EEP 539 Final Project
## HHL Algorithm Implementation to Solve Systems of Linear Equations
### Ryan Chu and Aditya Bagchi

### Summary of work

For our final project, we chose to implement the Harrow–Hassidim–Lloyd algorithm, which is a method to solve equations of the form Ax = b. The implementation was written using both qiskit and workbench for 2x2 matrices.

The algorithm encodes the right-hand side vector b as a quantum state, then uses Quantum Phase Estimation (QPE) to extract the eigenvalues of A into a clock register. An ancilla qubit is then rotated by an angle proportional to 1/λ for each eigenvalue λ — this is the key step that performs the matrix inversion. After measuring the ancilla and post-selecting on the success outcome, inverse QPE is applied to uncompute the clock register, leaving the b-register in a state whose amplitudes are proportional to the solution x = A⁻¹b. The solution is recovered by reading out the b-register amplitudes.

There are several conditions that must be satisfied for this algorithm to work. Firstly, the matrix A must be Hermitian, meaning, A must be equal to its conjugate transpose. In our implementation, the eigenvalues must also map to integer bins in the clock register, which requires the eigenvalue ratio to be rational and expressible within the available clock qubits. 

### Running our scripts
Each library implementation is separated into three files: functions, tests, and a main script. In each main script, you are able to define the problem's parameters:

```
A     = np.array([[3, 1], [1, 3]])
b_vec = np.array([1, 2], dtype=float)
n_l   = 4   # clock qubits
```

Note, the A matrix must be hermitian (equals its own conjugate transpose). If the eigenvalues of the A matrix are irrational, increasing the number of clock qubits will increase precision and thus the accuracy of the final solution. 

After running main in a terminal, e.g.
```
python qiskit_main.py
```
you should see an output showing the computed quantum solution, as well as the classically computed solution for comparison.

### If we had more time...

The HHL algorithm utilizes some key parameters: time evolution t and a scaling factor C. In our implementation, both are set using the eigenvalues of A computed classically. The evolution time t is chosen so that the eigenvalues map exactly to integer bins in the clock register, and C is set to 0.9 · λ_min to ensure the ancilla rotation angles are valid. This works for demonstration purposes but is circular — if you could compute the eigenvalues classically, you are already most of the way to solving the system classically.

In a true quantum implementation, neither t nor C would be known in advance. The evolution time t would instead be chosen based on known bounds on the spectrum of A, such as bounds derived from the structure of the problem (e.g. Gershgorin circle theorem for diagonally dominant matrices). The scaling factor C would similarly be set using a lower bound on λ_min rather than its exact value. The ancilla rotation angles would then be computed coherently as a quantum arithmetic circuit acting on the clock register — rather than a classical loop over known eigenvalues — so that the correct 1/λ inversion is applied to each eigenvalue without ever measuring or classically knowing which bin the system is in. This coherent implementation is the core of what makes HHL a genuine quantum algorithm, and also what makes it significantly harder to build in practice than our simulation suggests.

In addition, computing the unitary e ^ {-i A t} is a nontrivial task. For our implementation, we used scipy.linalg.expm(), which uses the Pade approximation to compute the matrix exponential. This would actually completely eliminate any computational speedup obtained from the HHL algorithm, since this approximation runs at O(N^3).

There is indeed a solution to this problem that uses quantum gates known as the Trotter-Suzuki formula that approximates the hamiltonian simulation: U=exp(−iHt). Both qiskit and workbench seems to contain functions that implement this formula. Unfortunately, we did not have enough time to understand and make use of these functions. While Trotterization introduces its own error that needs to be controlled, and the gate count scales with the sparsity and precision required, this would probably be priority #1 in terms of achieving a truely quantum solution. 
