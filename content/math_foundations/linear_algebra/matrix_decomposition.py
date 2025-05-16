def get_content():
    return {
        "section": [
            {
                "title": "Matrix Decomposition Fundamentals",
                "description": """
                <p>Matrix decomposition is a fundamental concept in linear algebra that breaks down a matrix into a product of simpler matrices.</p>
                <p>Key types of decomposition:</p>
                <ul>
                    <li>Eigendecomposition (Spectral Decomposition)</li>
                    <li>Singular Value Decomposition (SVD)</li>
                    <li>QR Decomposition</li>
                    <li>LU Decomposition</li>
                </ul>
                """,
                "formula": "$$A = Q\Lambda Q^T \text{ (Eigendecomposition)}$$",
                "example": "In principal component analysis (PCA), we use eigendecomposition to find the principal components of a dataset."
            },
            {
                "title": "Singular Value Decomposition",
                "description": """
                <p>SVD is a powerful matrix decomposition method that generalizes eigendecomposition to non-square matrices.</p>
                <p>Applications:</p>
                <ul>
                    <li>Dimensionality Reduction</li>
                    <li>Image Compression</li>
                    <li>Recommendation Systems</li>
                    <li>Natural Language Processing</li>
                </ul>
                """,
                "formula": "$$A = U\Sigma V^T$$",
                "img": "img/svd_image_compression.png",
                "caption": "Visualization of SVD for image compression.",
                "example": "In image compression, SVD can be used to reduce the storage requirements while maintaining image quality."
            }
        ],
        "implementation": """
import numpy as np
from scipy.linalg import svd, qr, lu

class MatrixDecomposition:
    @staticmethod
    def eigendecomposition(A):
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    @staticmethod
    def svd_decomposition(A):
        # Compute SVD
        U, S, Vh = svd(A, full_matrices=False)
        
        return U, S, Vh
    
    @staticmethod
    def qr_decomposition(A):
        # Compute QR decomposition
        Q, R = qr(A)
        
        return Q, R
    
    @staticmethod
    def lu_decomposition(A):
        # Compute LU decomposition
        P, L, U = lu(A)
        
        return P, L, U

class SVDApplications:
    @staticmethod
    def low_rank_approximation(A, k):
        # Compute SVD
        U, S, Vh = svd(A, full_matrices=False)
        
        # Keep only k singular values
        S[k:] = 0
        
        # Reconstruct matrix
        A_approx = U @ np.diag(S) @ Vh
        
        return A_approx
    
    @staticmethod
    def image_compression(image, k):
        # Convert image to float
        image = image.astype(float)
        
        # Compute SVD for each color channel
        compressed_channels = []
        for channel in range(image.shape[2]):
            U, S, Vh = svd(image[:, :, channel], full_matrices=False)
            
            # Keep only k singular values
            S[k:] = 0
            
            # Reconstruct channel
            compressed_channel = U @ np.diag(S) @ Vh
            compressed_channels.append(compressed_channel)
        
        # Combine channels
        compressed_image = np.stack(compressed_channels, axis=2)
        
        return compressed_image.astype(np.uint8)
    
    @staticmethod
    def pca(X, n_components):
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Compute SVD
        U, S, Vh = svd(X_centered, full_matrices=False)
        
        # Project data onto principal components
        X_pca = X_centered @ Vh[:n_components].T
        
        return X_pca, Vh[:n_components]
        """,
        "interview_examples": [
            {
                "title": "Implementing SVD from Scratch",
                "description": "A common interview question about implementing SVD without using existing libraries.",
                "code": """
def power_iteration(A, num_iterations=100):
    # Initialize random vector
    b_k = np.random.rand(A.shape[1])
    
    for _ in range(num_iterations):
        # Compute matrix-vector product
        b_k1 = A @ b_k
        
        # Normalize
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    
    # Compute eigenvalue
    eigenvalue = b_k.T @ A @ b_k
    
    return eigenvalue, b_k

def svd_scratch(A, k):
    # Compute A^T A
    ATA = A.T @ A
    
    # Initialize arrays
    U = np.zeros((A.shape[0], k))
    S = np.zeros(k)
    V = np.zeros((A.shape[1], k))
    
    # Compute k singular values and vectors
    for i in range(k):
        # Power iteration for largest eigenvalue
        eigenvalue, eigenvector = power_iteration(ATA)
        
        # Store singular value and right singular vector
        S[i] = np.sqrt(eigenvalue)
        V[:, i] = eigenvector
        
        # Compute left singular vector
        U[:, i] = (A @ eigenvector) / S[i]
        
        # Deflate matrix
        ATA = ATA - eigenvalue * np.outer(eigenvector, eigenvector)
    
    return U, S, V.T
                """
            }
        ],
        "resources": [
            {
                "title": "Matrix Decomposition in Machine Learning",
                "url": "https://www.deeplearningbook.org/contents/linear_algebra.html"
            },
            {
                "title": "SVD Applications",
                "url": "https://en.wikipedia.org/wiki/Singular_value_decomposition#Applications"
            }
        ],
        "related_topics": [
            "Linear Algebra",
            "Principal Component Analysis",
            "Dimensionality Reduction",
            "Matrix Operations"
        ]
    } 