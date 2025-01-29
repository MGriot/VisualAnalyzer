import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
import matplotlib.pyplot as plt
from PIL import Image

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def total_distance(colors1, colors2, transform):
    transformed_colors1 = [transform_color(c, transform) for c in colors1]
    cost_matrix = np.array([[color_distance(c1, c2) for c2 in colors2] for c1 in transformed_colors1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()

def transform_color(color, transform):
    return np.dot(transform, np.append(color, 1))[:3]

def find_optimal_transform(colors1, colors2):
    initial_transform = np.eye(4)
    result = minimize(
        lambda t: total_distance(colors1, colors2, t.reshape(4, 4)),
        initial_transform.flatten(),
        method='L-BFGS-B'
    )
    return result.x.reshape(4, 4)

def plot_color_alignment(colors1, colors2, transformed_colors1):
    fig, ax = plt.subplots()
    ax.scatter(*zip(*colors1), color='blue', label='Original Colors')
    ax.scatter(*zip(*colors2), color='red', label='Target Colors')
    ax.scatter(*zip(*transformed_colors1), color='green', label='Transformed Colors')
    for i in range(len(colors1)):
        ax.plot([colors1[i][0], transformed_colors1[i][0]], [colors1[i][1], transformed_colors1[i][1]], 'k--')
    ax.legend()
    plt.xlabel('Red')
    plt.ylabel('Green')
    plt.title('Color Alignment')
    plt.show()

def apply_transform_to_image(image, transform):
    image_np = np.array(image)
    transformed_image_np = np.zeros_like(image_np)
    for i in range(image_np.shape[0]):
        for j in range(image_np.shape[1]):
            transformed_image_np[i, j] = transform_color(image_np[i, j], transform)
    return Image.fromarray(np.uint8(transformed_image_np))

if __name__ == "__main__":
    colors1 = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # Example colors
    colors2 = np.array([[254, 1, 1], [1, 254, 1], [1, 1, 254]])  # Example colors to align to
    optimal_transform = find_optimal_transform(colors1, colors2)
    transformed_colors1 = [transform_color(c, optimal_transform) for c in colors1]
    print("Optimal Transform Matrix:")
    print(optimal_transform)
    plot_color_alignment(colors1, colors2, transformed_colors1)
    
    # Apply the transformation to an entire image
    sample_image_path = input("Enter the path to the sample image: ")
    image = Image.open(sample_image_path)
    transformed_image = apply_transform_to_image(image, optimal_transform)
    transformed_image.show()
