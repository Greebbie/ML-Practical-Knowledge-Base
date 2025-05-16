def get_content():
    return {
        "section": [
            {
                "title": "Matrices: Foundations",
                "description": """
                <p>Matrices are rectangular arrays of numbers, symbols, or expressions arranged in rows and columns. They are fundamental to linear algebra and have widespread applications in machine learning.</p>
                <p>A matrix with m rows and n columns is called an m×n matrix. The notation A = [a<sub>ij</sub>] refers to a matrix A where a<sub>ij</sub> is the element in the ith row and jth column.</p>
                """,
                "formula": """$$A = 
                \\begin{bmatrix} 
                a_{11} & a_{12} & \\cdots & a_{1n} \\\\
                a_{21} & a_{22} & \\cdots & a_{2n} \\\\
                \\vdots & \\vdots & \\ddots & \\vdots \\\\
                a_{m1} & a_{m2} & \\cdots & a_{mn}
                \\end{bmatrix}$$"""
            },
            {
                "title": "Matrix Operations",
                "description": """
                <p>Key matrix operations include addition, scalar multiplication, matrix multiplication, transposition, and finding determinants and inverses.</p>
                <p><strong>Matrix Addition:</strong> If A and B are m×n matrices, their sum A+B is the m×n matrix obtained by adding corresponding elements.</p>
                <p><strong>Matrix Multiplication:</strong> If A is an m×n matrix and B is an n×p matrix, their product C = AB is an m×p matrix where each element c<sub>ij</sub> is the dot product of the ith row of A with the jth column of B.</p>
                <p><strong>Matrix Transposition:</strong> The transpose of an m×n matrix A is the n×m matrix A<sup>T</sup> obtained by swapping the rows and columns of A.</p>
                <p><strong>Matrix Determinant:</strong> The determinant of a square matrix is a scalar value that can be computed from its elements and provides information about the matrix's invertibility.</p>
                """,
                "formula": """
                $$\\text{Addition: } (A + B)_{ij} = A_{ij} + B_{ij}$$
                $$\\text{Scalar Multiplication: } (\\alpha A)_{ij} = \\alpha A_{ij}$$
                $$\\text{Matrix Multiplication: } (AB)_{ij} = \\sum_{k=1}^{n} A_{ik}B_{kj}$$
                $$\\text{Transpose: } (A^T)_{ij} = A_{ji}$$
                """
            },
            {
                "title": "Special Matrices",
                "description": """
                <p>Several types of special matrices are particularly important in machine learning:</p>
                <ul>
                    <li><strong>Identity Matrix (I):</strong> A square matrix with ones on the main diagonal and zeros elsewhere. It serves as the multiplicative identity.</li>
                    <li><strong>Diagonal Matrix:</strong> A matrix where all non-diagonal elements are zero.</li>
                    <li><strong>Symmetric Matrix:</strong> A square matrix equal to its transpose (A = A<sup>T</sup>).</li>
                    <li><strong>Orthogonal Matrix:</strong> A square matrix whose transpose equals its inverse (A<sup>T</sup>A = AA<sup>T</sup> = I).</li>
                    <li><strong>Positive Definite Matrix:</strong> A symmetric matrix where x<sup>T</sup>Ax > 0 for all non-zero vectors x.</li>
                </ul>
                """,
                "formula": """
                $$\\text{Identity Matrix: } I_n = 
                \\begin{bmatrix} 
                1 & 0 & \\cdots & 0 \\\\
                0 & 1 & \\cdots & 0 \\\\
                \\vdots & \\vdots & \\ddots & \\vdots \\\\
                0 & 0 & \\cdots & 1
                \\end{bmatrix}$$
                
                $$\\text{Diagonal Matrix: } D = 
                \\begin{bmatrix} 
                d_1 & 0 & \\cdots & 0 \\\\
                0 & d_2 & \\cdots & 0 \\\\
                \\vdots & \\vdots & \\ddots & \\vdots \\\\
                0 & 0 & \\cdots & d_n
                \\end{bmatrix}$$
                """
            },
            {
                "title": "Eigenvalues and Eigenvectors",
                "description": """
                <p>For a square matrix A, an eigenvector is a non-zero vector v such that Av is parallel to v. In other words, Av = λv for some scalar λ, which is called the eigenvalue.</p>
                <p>Eigendecomposition is the factorization of a matrix into its canonical form, revealing the underlying structure of the matrix through its eigenvalues and eigenvectors.</p>
                <p>For a symmetric matrix, eigendecomposition can be written as A = QΛQ<sup>T</sup>, where Q is an orthogonal matrix whose columns are the eigenvectors of A, and Λ is a diagonal matrix containing the eigenvalues.</p>
                """,
                "formula": """
                $$A\\mathbf{v} = \\lambda \\mathbf{v}$$
                
                $$\\text{Eigendecomposition: } A = Q\\Lambda Q^{-1}$$
                
                $$\\text{For symmetric matrices: } A = Q\\Lambda Q^T$$
                """
            }
        ],
        "implementation": """
import numpy as np

# Creating matrices
A = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

B = np.array([[9, 8, 7], 
              [6, 5, 4], 
              [3, 2, 1]])

# Matrix operations
addition = A + B
subtraction = A - B
scalar_mult = 2 * A
matrix_mult = A @ B  # or np.matmul(A, B)
transpose = A.T

# Special matrices
identity = np.eye(3)
diagonal = np.diag([1, 2, 3])

# Determinant and inverse
det_A = np.linalg.det(A)

# Not every matrix is invertible
try:
    inv_A = np.linalg.inv(A)
except np.linalg.LinAlgError:
    print("Matrix is not invertible")

# Eigenvalues and eigenvectors
# For a better example, let's use a symmetric matrix
S = np.array([[4, 2, 2], 
              [2, 5, 1], 
              [2, 1, 6]])

eigenvalues, eigenvectors = np.linalg.eig(S)

# Verify eigendecomposition: S = Q Λ Q^T
Q = eigenvectors
Lambda = np.diag(eigenvalues)
reconstructed_S = Q @ Lambda @ Q.T

# Check if matrices are nearly equal (considering numerical precision)
is_equal = np.allclose(S, reconstructed_S)
""",
        "interview_examples": [
            {
                "title": "Matrix Multiplication Properties",
                "description": "Explain the key properties of matrix multiplication and why it is not commutative in general.",
                "code": """
# Matrix multiplication properties:
# 1. Not commutative: AB ≠ BA (in general)
# 2. Associative: (AB)C = A(BC)
# 3. Distributive: A(B+C) = AB+AC

# Example showing non-commutativity
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

AB = A @ B
BA = B @ A

print(f"AB = {AB}")  # [[19, 22], [43, 50]]
print(f"BA = {BA}")  # [[23, 34], [31, 46]]
print(f"AB == BA: {np.array_equal(AB, BA)}")  # False
"""
            },
            {
                "title": "Finding Eigenvalues and Eigenvectors",
                "description": "How do you find eigenvalues and eigenvectors of a matrix, and why are they important in machine learning?",
                "code": """
# To find eigenvalues and eigenvectors:
# 1. For eigenvalues, solve the characteristic equation det(A - λI) = 0
# 2. For each eigenvalue λ, find eigenvector v by solving (A - λI)v = 0

# In machine learning, eigenvalues and eigenvectors are important for:
# - PCA (Principal Component Analysis): finding directions of maximum variance
# - Covariance matrices: understanding data correlation structure
# - Spectral methods: clustering, dimension reduction
# - Understanding linear transformations

import numpy as np

# Example: Finding eigenvalues and eigenvectors of a covariance matrix
data = np.random.randn(100, 2)  # Generate random 2D data
cov = np.cov(data, rowvar=False)  # Compute covariance matrix

eigenvalues, eigenvectors = np.linalg.eigh(cov)  # Use eigh for symmetric matrices

# Sort by eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Principal eigenvalue: {eigenvalues[0]}")
print(f"Principal component: {eigenvectors[:, 0]}")
"""
            }
        ],
        "resources": [
            {"title": "Linear Algebra and Its Applications by Gilbert Strang", "url": "https://math.mit.edu/~gs/linearalgebra/"},
            {"title": "3Blue1Brown: Essence of Linear Algebra", "url": "https://www.3blue1brown.com/topics/linear-algebra"},
            {"title": "The Matrix Cookbook", "url": "https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf"}
        ],
        "related_topics": [
            "Linear Transformations", "Vector Spaces", "Matrix Decompositions", "Singular Value Decomposition", "Neural Network Weight Matrices"
        ]
    } 