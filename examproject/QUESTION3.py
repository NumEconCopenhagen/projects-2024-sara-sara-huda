import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

f = lambda x: x[0] * x[1]

rng = np.random.default_rng(2024)
X = rng.uniform(size=(50, 2))
F = np.array([f(x) for x in X])

grid_x = np.linspace(0, 1, 10)
grid_y = np.linspace(0, 1, 10)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
grid = np.array([grid_xx.flatten(), grid_yy.flatten()]).T
values = np.array([f(point) for point in grid])

interpolator = RegularGridInterpolator((grid_x, grid_y), values.reshape((10, 10)))

def find_points(X, y):
    A, B, C, D = None, None, None, None
    min_dist_A, min_dist_B, min_dist_C, min_dist_D = float('inf'), float('inf'), float('inf'), float('inf')
    
    for point in X:
        x1, x2 = point
        y1, y2 = y
        dist = np.sqrt((x1 - y1)**2 + (x2 - y2)**2)
        
        if x1 > y1 and x2 > y2 and dist < min_dist_A:
            A, min_dist_A = point, dist
        if x1 > y1 and x2 < y2 and dist < min_dist_B:
            B, min_dist_B = point, dist
        if x1 < y1 and x2 < y2 and dist < min_dist_C:
            C, min_dist_C = point, dist
        if x1 < y1 and x2 > y2 and dist < min_dist_D:
            D, min_dist_D = point, dist
    
    return A, B, C, D

def barycentric_coordinates(y, A, B, C):
    y1, y2 = y
    A1, A2 = A
    B1, B2 = B
    C1, C2 = C

    denominator = (B2 - C2) * (A1 - C1) + (C1 - B1) * (A2 - C2)
    r1 = ((B2 - C2) * (y1 - C1) + (C1 - B1) * (y2 - C2)) / denominator
    r2 = ((C2 - A2) * (y1 - C1) + (A1 - C1) * (y2 - C2)) / denominator
    r3 = 1 - r1 - r2

    return r1, r2, r3

# Approximate f(y)
# We used ChatGPT to help us construct this function, which was important for accurately approximating f(y)
def approximate_f_y(y, X, F):
    A, B, C, D = find_points(X, y)
    
    if A is None or B is None or C is None or D is None:
        return np.nan, 'none'
    
    r_ABC = barycentric_coordinates(y, A, B, C)
    r_CDA = barycentric_coordinates(y, C, D, A)

    inside_ABC = all(0 <= r <= 1 for r in r_ABC)
    inside_CDA = all(0 <= r <= 1 for r in r_CDA)

    if inside_ABC:
        f_A, f_B, f_C = f(A), f(B), f(C)
        approximation = r_ABC[0] * f_A + r_ABC[1] * f_B + r_ABC[2] * f_C
        triangle = 'ABC'
    elif inside_CDA:
        f_C, f_D, f_A = f(C), f(D), f(A)
        approximation = r_CDA[0] * f_C + r_CDA[1] * f_D + r_CDA[2] * f_A
        triangle = 'CDA'
    else:
        approximation = np.nan
        triangle = 'none'

    return approximation, triangle

def plot_points_and_triangles(X, y, A, B, C, D):
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], label='Points in X', color='blue')
    plt.scatter(y[0], y[1], color='red', label='y')

    if A is not None: plt.scatter(A[0], A[1], color='green', label='A')
    if B is not None: plt.scatter(B[0], B[1], color='orange', label='B')
    if C is not None: plt.scatter(C[0], C[1], color='purple', label='C')
    if D is not None: plt.scatter(D[0], D[1], color='brown', label='D')

    if A is not None and B is not None and C is not None:
        plt.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], color='green', linestyle='dashed', label='Triangle ABC')
    if C is not None and D is not None and A is not None:
        plt.plot([C[0], D[0], A[0], C[0]], [C[1], D[1], A[1], C[1]], color='purple', linestyle='dashed', label='Triangle CDA')

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Points and Triangles')
    plt.legend()
    plt.grid(True)
    plt.show()

y = rng.uniform(size=(2,))
A, B, C, D = find_points(X, y)
plot_points_and_triangles(X, y, A, B, C, D)

def example_barycentric_and_containment(y, A, B, C, D):
    r_ABC = barycentric_coordinates(y, A, B, C)
    r_CDA = barycentric_coordinates(y, C, D, A)

    inside_ABC = all(0 <= r <= 1 for r in r_ABC)
    inside_CDA = all(0 <= r <= 1 for r in r_CDA)

    if inside_ABC:
        containing_triangle = 'ABC'
    elif inside_CDA:
        containing_triangle = 'CDA'
    else:
        containing_triangle = 'none'

    print(f"Barycentric coordinates w.r.t. triangle ABC: r1={r_ABC[0]:.3f}, r2={r_ABC[1]:.3f}, r3={r_ABC[2]:.3f}")
    print(f"Barycentric coordinates w.r.t. triangle CDA: r1={r_CDA[0]:.3f}, r2={r_CDA[1]:.3f}, r3={r_CDA[2]:.3f}")
    print(f"The point y is inside triangle: {containing_triangle}")

example_barycentric_and_containment(y, A, B, C, D)

def example_compute_approximation(y, X, F):
    true_value = f(y)
    approximation, triangle = approximate_f_y(y, X, F)
    print(f"True value of f(y): {true_value:.3f}")
    print(f"Approximated value of f(y): {approximation:.3f}")
    print(f"Point y is inside triangle: {triangle}")

example_compute_approximation(y, X, F)

def example_for_all_points_in_Y(Y, X, F):
    results = []
    for y in Y:
        true_value = f(y)
        approximation, triangle = approximate_f_y(y, X, F)
        results.append((y, true_value, approximation, triangle))

    
    for result in results:
        y, true_value, approximation, triangle = result
        print(f"Point y: {y}")
        print(f"  True value of f(y): {true_value:.3f}")
        print(f"  Approximated value of f(y): {approximation:.3f}")
        print(f"  Point y is inside triangle: {triangle}\n")

Y = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.5, 0.5)]
example_for_all_points_in_Y(Y, X, F)