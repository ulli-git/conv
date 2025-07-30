import numpy as np

def convolution(matrix: np.ndarray, kernel:np.ndarray, padding: int = 0) -> np.ndarray:
    result = []
    kernel_height, kernel_width = kernel.shape
    matrix_height, matrix_width = matrix.shape
    if (matrix_height != matrix_width) or (kernel_height != kernel_width):
        raise AssertionError #ValueError
    if (matrix_height < kernel_height) or (matrix_width < kernel_width):
        raise AssertionError #ValueError
    if not (kernel_height % 2 == 1):
        raise AssertionError #ValueError
    
    padding_size = kernel_height + ( kernel_height - 1 )
    if (matrix_height <= padding_size):
        padding_width = int(( padding_size - matrix_height ) / 2)
        matrix_padded = np.pad(matrix, pad_width=padding_width, mode='constant', constant_values=0)
    else:
        delta = matrix_height % kernel_height
        if delta == 0:
            matrix_padded = matrix
        else:
            padding_width = int(delta / 2)
            matrix_padded = np.pad(matrix, pad_width=padding_width, mode='constant', constant_values=0)
    
    for row_i in range(0, matrix_height, 1):
        row_result = []
        submatrix_row = slice(row_i, row_i + kernel_height)
        for column_i in range(0, matrix_width, 1):
            submatrix_column = slice(column_i, column_i + kernel_width)
            submatrix_indexes = submatrix_row, submatrix_column
            z = np.sum(matrix_padded[submatrix_indexes] * kernel)
            row_result.append(z)
        result.append(row_result)
    return np.array(result)

if __name__ == '__main__':
    np.random.seed(0)
    matrix = np.random.randint(0, 255, (32,32))
    kernel = np.ones((3,3)) * -1
    kernel[1,1] = 9
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    #print(matrix)
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=float)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)   
    #print(kernel)
    #([[ 7, 11, 11],
    #  [17, 25, 23],
    #  [19, 29, 23]]
    #matrix = np.array([[1, 2], [3, 4]], dtype=float)
    #kernel = np.array([[1, 0], [0, 0], [0, 1]], dtype=float)
    #print(convolution(matrix, kernel))
    np.random.seed(0)
    matrix = np.random.randint(0, 255, (32,32))
    kernel = np.ones((3,3)) * -1
    kernel[1,1] = 9
    print(convolution(matrix, kernel))