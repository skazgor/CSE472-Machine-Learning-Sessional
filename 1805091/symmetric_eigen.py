import numpy as np
def symmetric_invertible_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        if(i==0):
            A[i][i] = np.random.randint(1,20)
        else:
            total_sum_left_of_diagonal = np.sum(A[i,:i])
            A[i][i] = np.random.randint(total_sum_left_of_diagonal+1,total_sum_left_of_diagonal+20)
        max_sum = A[i][i] - np.sum(A[i,:i])
        for j in range(i+1,n):
            A[i][j] = np.random.randint(max_sum-20,max_sum)
            max_sum -= A[i][j]
            A[j][i] = A[i][j]
    return A


shouldPrint = True
n=int(input("Enter the dimenstion of the matrix: "))
if n<2:
    print("The dimension of the matrix should be greater than 1")
    exit()
if n>10:
    shouldPrint = False
    print("The dimension of the matrix is too large to be printed")


# matrix initialization
matrix=symmetric_invertible_matrix(n)



print("The random symmetric matrix is: ")

if shouldPrint:
    print(matrix)


# eigenvalues and eigenvectors of the matrix
eigenvalues, eigenvectors = np.linalg.eig(matrix)


if shouldPrint:
    print("The eigenvalues are: ")
    print(eigenvalues)

if shouldPrint:
    print("The eigenvectors are: ")
    print(eigenvectors)


# diagonal matrix
diag_matrix = np.diag(eigenvalues)


# reconstructed matrix
reconstructed_matrix = np.dot(np.dot(eigenvectors,diag_matrix),np.linalg.inv(eigenvectors))

if shouldPrint:
    print("The reconstructed matrix is: ")
    print(reconstructed_matrix)

# check if the matrix is reconstructed correctly
if(np.allclose(matrix,reconstructed_matrix)):
    print("The matrix is reconstructed correctly")