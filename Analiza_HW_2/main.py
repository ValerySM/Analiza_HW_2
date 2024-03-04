import numpy as np

def check_dominant_diagonal(matrix):
    # Check if there is a dominant diagonal in the matrix
    diagonal_elements = np.diag(matrix)
    sum_of_abs_row_elements = np.sum(np.abs(matrix), axis=1) - np.abs(diagonal_elements)

    if np.all(diagonal_elements > sum_of_abs_row_elements):
        print("The matrix has a dominant diagonal.")
        return True
    else:
        print("The matrix does not have a dominant diagonal.")
        return False

def jacobi_method(coefficients, constants, initial_guess, tolerance=1e-10, max_iterations=1000):
    n = len(coefficients)
    x = initial_guess.copy()
    iteration = 0

    if not check_dominant_diagonal(coefficients):
        return None

    while iteration < max_iterations:
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(coefficients[i, :i], x_old[:i]) + np.dot(coefficients[i, i+1:], x_old[i+1:])
            x[i] = (constants[i] - sigma) / coefficients[i, i]

        if np.max(np.abs(x - x_old)) < tolerance:
            print(f"Jacobi Method: Converged in {iteration + 1} iterations.")
            return x

        iteration += 1

    print("Jacobi method did not converge within the specified number of iterations.")
    return None

def gauss_seidel_method(coefficients, constants, initial_guess, tolerance=1e-10, max_iterations=1000):
    n = len(coefficients)
    x = initial_guess.copy()
    iteration = 0

    if not check_dominant_diagonal(coefficients):
        return None

    while iteration < max_iterations:
        x_old = x.copy()
        for i in range(n):
            sigma = np.dot(coefficients[i, :i], x[:i]) + np.dot(coefficients[i, i+1:], x_old[i+1:])
            x[i] = (constants[i] - sigma) / coefficients[i, i]

        if np.max(np.abs(x - x_old)) < tolerance:
            print(f"Gauss-Seidel Method: Converged in {iteration + 1} iterations.")
            return x

        iteration += 1

    print("Gauss-Seidel method did not converge within the specified number of iterations.")
    return None

if __name__ == "__main__":
    # Example usage with a 4x4 square matrix
    coefficients = np.array([[4, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]], dtype=float)
    constants = np.array([15, 10, 10, 10], dtype=float)
    initial_guess = np.zeros_like(constants)

    jacobi_result = jacobi_method(coefficients, constants, initial_guess, tolerance=0.001)
    print("Jacobi Result:", jacobi_result)

    gauss_seidel_result = gauss_seidel_method(coefficients, constants, initial_guess, tolerance=0.001)
    print("Gauss-Seidel Result:", gauss_seidel_result)
