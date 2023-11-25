import numpy as np
def random_invertible_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i][i] = np.random.randint(1,20)
        max_sum = A[i][i]
        for j in range(n):
            if i!=j:
                A[i][j] = np.random.randint(max_sum-20,max_sum)
                max_sum -= A[i][j]
    return A




shouldPrint = True
n=int(input("Enter the dimenstion of the matrix: "))
if(n<2):
    print("The dimension of the matrix should be greater than 1")
    exit()
if(n>10):
    shouldPrint = False

# matrix initialization
matrix=random_invertible_matrix(n)



print("The random matrix is: ")
if shouldPrint:
    print(matrix)
else:
    print("The dimension of the matrix is too large to be printed")


# eigenvalues and eigenvectors of the matrix
eigenvalues, eigenvectors = np.linalg.eig(matrix)



print("The eigenvalues are: ")
if shouldPrint:
    print(eigenvalues)
else:
    print("The dimension of the matrix is too large to be printed")

print("The eigenvectors are: ")
if shouldPrint:
    print(eigenvectors)
else:
    print("The dimension of the matrix is too large to be printed")


# diagonal matrix
diag_matrix = np.diag(eigenvalues)


# reconstructed matrix
reconstructed_matrix = np.dot(np.dot(eigenvectors,diag_matrix),np.linalg.inv(eigenvectors))

print("The reconstructed matrix is: ")
if shouldPrint:
    print(reconstructed_matrix)
else:
    print("The dimension of the matrix is too large to be printed")


# check if the matrix is reconstructed correctly
if(np.allclose(matrix,reconstructed_matrix)):
    print("The matrix is reconstructed correctly")