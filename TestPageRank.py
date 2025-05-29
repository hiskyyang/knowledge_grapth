import numpy as np

M = np.array([
    [0,   0,   1, 0,   0],
    [0.5, 0,   0, 0,   1],
    [0.5, 0.5, 0, 0.5, 0],
    [0,   0.5, 0, 0,   0],
    [0,   0,   0, 0.5, 0]
])

V = np.full((5, 1), 1 / 5)
threshold = 0.01
iteration = 0
while True:
    iteration += 1
    V_next = M @ V
    # Check convergence
    if np.all(np.abs(V_next - V) < threshold):
        break
    V = V_next

print(f"PageRank vector after {iteration} iterations:")
print(V_next)