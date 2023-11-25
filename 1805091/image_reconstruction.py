import numpy as np
import matplotlib.pyplot as plt
import cv2

def low_rank_approximation(A, k):
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
    diagonal_matrix = np.diag(Sigma[:k])
    A_k = np.dot(U[:, :k], np.dot(diagonal_matrix, Vt[:k, :]))
    return A_k

def choose_smallest_k(A, threshold):
    U, Sigma, Vt = np.linalg.svd(A, full_matrices=False)
    total_sum = np.sum(Sigma)

    cumulative_sum = np.cumsum(Sigma)
    cumulative_percentage_sum = cumulative_sum / total_sum

    k= np.argmax(cumulative_percentage_sum >= threshold)+1

    return k


# Read the grayscale image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)


# Perform Singular Value Decomposition
U, Sigma, Vt = np.linalg.svd(image, full_matrices=False)

threshold = 0.65
k_smallest = choose_smallest_k(image, threshold)



# Vary the value of k and plot the resultant k-rank approximations
# first 10 values of k is spaced by 5. starting from 1. then spaced by 50 to the end
# rank
k= min(image.shape)
k_values = np.hstack((np.arange(1, min(k,50), 5), np.arange(100, k, 50)))
k_values= np.hstack((k_values, k_smallest))

image_rows=k_values.size//4
if(k_values.size%4!=0):
    image_rows+=1

plt.figure()
for i, rank in enumerate(k_values, 1):
    A_k = low_rank_approximation(image, rank)

    plt.subplot(image_rows, 4 , i)
    plt.title(f'Rank = {rank}')
    plt.imshow(A_k, cmap='gray')
    plt.axis('off')

plt.tight_layout(w_pad=0.1, h_pad=0.2)
plt.show()

