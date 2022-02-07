from PIL import Image
import numpy
import math


def from_rgb_in_ycbcr(pixels_array: numpy.ndarray):
    def pixel_in_ycbcr(pixel: numpy.ndarray):
        r, g, b = pixel.astype(float)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return numpy.clip(numpy.array((y, cb, cr)), 0, 255).astype(numpy.uint8)

    size_arr = pixels_array.shape
    ycbcr_array = numpy.zeros(size_arr).astype(numpy.uint8)
    for i in range(size_arr[0]):
        for j in range(size_arr[1]):
            ycbcr_array[i][j] = pixel_in_ycbcr(pixels_array[i][j])
    return ycbcr_array


def from_ycbcr_in_rgb(pixels_array: numpy.ndarray):
    def pixel_in_rgb(pixel: numpy.ndarray):
        y, cb, cr = pixel.astype(float)
        r = y + 1.402 * (cr - 128)
        g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
        b = y + 1.772 * (cb - 128)
        return numpy.clip(numpy.array((r, g, b)), 0, 255).astype(numpy.uint8)

    size_arr = pixels_array.shape
    rgb_array = numpy.zeros(size_arr).astype(numpy.uint8)
    for i in range(size_arr[0]):
        for j in range(size_arr[1]):
            rgb_array[i][j] = numpy.clip(pixel_in_rgb(pixels_array[i][j]), 0, 255).astype(numpy.uint8)
    return rgb_array


def block_8_8_from_matrix(matrix: numpy.ndarray, row: int, column: int):
    """Извлечение блока 8 на 8 из матрицы"""  # протестировано
    block = numpy.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            block[i][j] = matrix[row + i][column + j]
    return block


def extract_2d_matrix(matrix: numpy.ndarray, index_2d_matrix: int) -> numpy.ndarray:
    """Извлекаем из трехмерной матрицы прямоугольною (строку столбцов)"""  #
    size_matrix = matrix.shape
    matrix_2d = numpy.zeros((size_matrix[0], size_matrix[1]), dtype=float)
    for i in range(size_matrix[0]):
        for j in range(size_matrix[1]):
            matrix_2d[i][j] = matrix[i][j][index_2d_matrix]
    return matrix_2d


def replace_2d_matrix(matrix: numpy.ndarray, matrix_2d: numpy.ndarray, index_2d_matrix: int):
    size_matrix = matrix_2d.shape
    for i in range(size_matrix[0]):
        for j in range(size_matrix[1]):
            matrix[i][j][index_2d_matrix] = matrix_2d[i][j]


def block_8_8_in_2d_matrix(matrix: numpy.ndarray, block_8_8, row: int, column: int):
    for i in range(8):
        for j in range(8):
            matrix[row + i][column + j] = block_8_8[i][j]


def do_operation_on_blocks(image_arr: numpy.ndarray, operation_on_block):
    """Преобразование матрицы"""
    size_array = image_arr.shape
    new_image_array = numpy.zeros_like(image_arr).astype(
        float)

    cb_array, cr_array = extract_2d_matrix(image_arr, 1), extract_2d_matrix(image_arr, 2)
    for i in range(0, size_array[0], 8):
        for j in range(0, size_array[1], 8):
            block_8_8_in_2d_matrix(cb_array, operation_on_block(block_8_8_from_matrix(cb_array, i, j)), i, j)
            block_8_8_in_2d_matrix(cr_array, operation_on_block(block_8_8_from_matrix(cr_array, i, j)), i, j)

    replace_2d_matrix(new_image_array, extract_2d_matrix(image_arr, 0), 0)
    replace_2d_matrix(new_image_array, cb_array, 1)
    replace_2d_matrix(new_image_array, cr_array, 2)

    return new_image_array.astype(int)


def create_dct_matrix():
    """Создание ДКП матрицы"""
    def dct_elem(i, j):
        """Создание коэффициента ДКП"""
        if i == 0:
            return 1 / math.sqrt(8)
        else:
            return math.sqrt(2 / 8) * math.cos((2 * j + 1) * i * math.pi / (2. * 8))

    dct_matrix = numpy.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            dct_matrix[i][j] = dct_elem(i, j)
    return dct_matrix


def do_dct(image_arr):
    def dct_on_block(matrix):
        dct_matrix = create_dct_matrix()
        return numpy.dot(dct_matrix, matrix).dot(numpy.transpose(dct_matrix))

    return do_operation_on_blocks(image_arr, dct_on_block)


def do_inverse_dct(image_arr):
    def dct_inverse_on_block(matrix):
        dct_matrix = create_dct_matrix()
        return numpy.dot(numpy.transpose(dct_matrix), matrix).dot(dct_matrix)

    return do_operation_on_blocks(image_arr, dct_inverse_on_block)


def create_rounding_matrix(coefficient):
    """
    Создание матрицы округления/квантования/адаптивная матрица
    :param coefficient: коэффициент сжатия
    :return:
    """
    quantization_matrix = [[None] * 8 for i in range(8)]
    for i in range(8):
        for j in range(8):
            quantization_matrix[i][j] = float(1 + (1 + i + j) * coefficient)
    return quantization_matrix


def do_quantization(image_arr, quantization_coefficient):
    def dividing_elements_of_matrix(matrix, quantization_matrix):
        """Поэлементное деление"""
        dividings_matrix = [[None] * 8 for i in range(8)]
        for i in range(8):
            for j in range(8):
                dividings_matrix[i][j] = matrix[i][j] / quantization_matrix[i][j]

        return dividings_matrix

    def do_quantization_on_block(matrix):
        rounding_matrix = create_rounding_matrix(quantization_coefficient)
        return dividing_elements_of_matrix(matrix, rounding_matrix)

    return do_operation_on_blocks(image_arr, do_quantization_on_block)

def do_inverse_quantization(image_arr, quantization_coefficient):
    def multiply_elements_of_matrix(matrix, quantization_matrix):
        """Поэлементное умножение"""
        dividings_matrix = [[None] * 8 for i in range(8)]
        for i in range(8):
            for j in range(8):
                dividings_matrix[i][j] = int((matrix[i][j] * quantization_matrix[i][j]))
        return dividings_matrix

    def do_inverse_quantization_on_block(matrix):
        rounding_matrix = create_rounding_matrix(quantization_coefficient)
        return multiply_elements_of_matrix(matrix, rounding_matrix)
    return do_operation_on_blocks(image_arr, do_inverse_quantization_on_block)

def create_matrix_to_multiple_8(image_array):
    """Расширение массива пикселей до кратного восьми по ширине и высоте"""
    size_array_old = list(image_array.shape)
    size_array_new = [size_array_old[0], size_array_old[1], 3]

    if size_array_old[0] % 8 != 0:
        size_array_new[0] += 8 - (size_array_old[0] % 8)
    if size_array_old[1] % 8 != 0:
        size_array_new[1] += 8 - (size_array_old[1] % 8)

    new_image_array = numpy.zeros((size_array_new[0], size_array_new[1], 3), dtype=numpy.uint8)
    # копирование изображения в больший массив
    for i in range(size_array_old[0]):
        for j in range(size_array_old[1]):
            for k in range(size_array_old[2]):
                new_image_array[i][j][k] = image_array[i][j][k]

    # копирование крайних массивов в новые ячейки
    for i in range(size_array_old[0]):
        for j in range(size_array_old[1], size_array_new[1]):
            for k in range(3):
                new_image_array[i][j][k] = image_array[i][-1][k]
    for i in range(size_array_old[0], size_array_new[0]):
        new_image_array[i] = new_image_array[size_array_old[0] - 1]
    return new_image_array


class Jpeg:
    def __init__(self, name_file, quantization_coefficient: int = 2):
        self.quantization_coefficient = quantization_coefficient # задание коэффициента квантования
        self.old_images_array = numpy.asarray(Image.open(name_file)) # создание массива пикселей текущего изображения
        self.old_images_array_multiple_8 = create_matrix_to_multiple_8(self.old_images_array) # создание массива пикселей
        # текущего изображения, но уже с изменёнными длиной и шириной в случае не кратности 8
        self.images_array_after_jpg = None  # массив пикселей нового изображения в rgb
        self._in_from_jpg_array() # вызов метода начала работы над изображением

    def _in_from_jpg_array(self):
        # прямое преобразование
        ycbcr_array = from_rgb_in_ycbcr(self.old_images_array_multiple_8)
        dct_array = do_dct(ycbcr_array)
        quantization_array = do_quantization(dct_array, self.quantization_coefficient)

        # обратное преобразование
        inverse_quantization_array = do_inverse_quantization(quantization_array, self.quantization_coefficient)
        inverse_dct_array = do_inverse_dct(inverse_quantization_array)
        result_image_array = from_ycbcr_in_rgb(inverse_dct_array)
        result_image_array_like_old = numpy.zeros_like(self.old_images_array)
        for i in range(self.old_images_array.shape[0]):
            for j in range(self.old_images_array.shape[1]):
                result_image_array_like_old[i][j] = result_image_array[i][j]

        self.images_array_after_jpg = result_image_array_like_old

    def save_compressed_image(self, name_file):
        Image.fromarray(self.images_array_after_jpg).save(name_file + '.png')

    def save_image_difference(self, name_file: str = 'differences_image', lightening_coefficient=3):
        def lightening(pixel, degree=3):
            return round(((pixel / 256.) ** (1 / degree)) * 256)

        size_arr = self.old_images_array.shape
        differences_image = numpy.zeros(size_arr).astype(numpy.uint8)
        for i in range(size_arr[0]):
            for j in range(size_arr[1]):
                for k in range(size_arr[2]):
                    differences_image[i][j][k] = numpy.clip(
                        lightening(abs(
                            int(self.old_images_array[i][j][k]) -
                            int(self.images_array_after_jpg[i][j][k])
                        ), lightening_coefficient), 0, 255
                    ).astype(numpy.uint8)
        Image.fromarray(differences_image).save(name_file + '.png')
