import logging
import concurrent.futures
from enum import Enum
from time import perf_counter

import numpy as np

""" Logging setup start """
str_logging_format = "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(str_logging_format)

file_handler = logging.FileHandler("solver.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
""" Logging setup end """


EPSILON = 0.00001


class MatrixOpEnum(Enum):
    A, B, C = 'A', 'B', 'C'

    def __str__(self):
        return self.value


class MatrixOperation:
    def __init__(self, operation: MatrixOpEnum, i=None, j=None, k=None):
        self.op = operation
        self.i = i
        self.k = k
        self.j = j

    def __str__(self):
        return f'{self.op}_{self.k}_{self.j if self.j is not None else "-"}_{self.i}'


class LinearEquationSolver:
    def __init__(self, input_file_path: str, output_file_path: str):
        logger.info(f"Created {self.__class__.__name__} instance.")

        self.input_path = input_file_path
        self.n = None
        self.eps = EPSILON
        self.input_matrix = None
        self.FNF = None
        self.m_dict = None
        self.n_dict = None
        self.output_matrix = None
        self.output_str = None
        self.output_path = output_file_path

        self.get_matrix_from_file()

    def get_matrix_from_file(self):
        with open(self.input_path, mode='r') as file:
            lines = file.readlines()
            self.n = int(lines[0])
            last_col = np.array([float(x) for x in lines[-1].split()])
            rows = [[float(x) for x in lines[i].split()] for i in range(1, len(lines) - 1)]
            matrix = np.c_[np.array(rows, dtype=np.float32), last_col]

            # logger.debug(f"extended matrix: {matrix}")

            self.input_matrix = matrix

    def find_FNF(self):
        logger.info("Finding FNF - START")
        self.FNF = []
        for i in range(1, self.n):
            # F_Ai
            F_Ai = [MatrixOperation(MatrixOpEnum.A, i=i, k=x)
                    for x in range(i + 1, self.n + 1)]

            # logger.debug(f"F_A{i}: {[str(x) for x in F_Ai]}")

            # F_Bi
            F_Bi = [MatrixOperation(MatrixOpEnum.B, i=i, j=y, k=x)
                    for x in range(i + 1, self.n + 1)
                    for y in range(i, self.n + 2)]

            # logger.debug(f"F_B{i}: {[str(x) for x in F_Bi]}")

            # F_Ci
            F_Ci = [MatrixOperation(MatrixOpEnum.C, i=i, j=y, k=x)
                    for x in range(i + 1, self.n + 1)
                    for y in range(i, self.n + 2)]

            # logger.debug(f"F_C{i}: {[str(x) for x in F_Ci]}")

            self.FNF.append(F_Ai)
            self.FNF.append(F_Bi)
            self.FNF.append(F_Ci)

        logger.info(f"FNF: {[[str(x) for x in Y] for Y in self.FNF]}")
        logger.info("Finding FNF - END")

    def gaussian_elimination_concurrent(self):
        logger.info("Performing gaussian elimination - START")
        t1 = perf_counter()

        self.m_dict = dict()
        self.n_dict = dict()
        self.output_matrix = self.input_matrix.copy()

        # Perform operations in FNF order
        for fnf_class in self.FNF:
            logger.debug(f"Performing operations concurrently from fnf_class: {[str(x) for x in fnf_class]}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(fnf_class)) as executor:
                executor.map(self.perform_operation, fnf_class)

        # Clear matrix of zero-close values
        self.output_matrix = np.where(abs(self.output_matrix) > self.eps,
                                      self.output_matrix,
                                      np.zeros_like(self.output_matrix))

        t2 = perf_counter()

        logger.info(f"Result of gaussian elimination:\n {self.output_matrix}")
        logger.info(f"Time of gaussian elimination: {t2-t1}ms")
        logger.info("Performing gaussian elimination - END")

    def perform_operation(self, operation: MatrixOperation):
        logger.debug(f"Performing operation: {str(operation)}")
        k = operation.k - 1
        j = operation.j - 1 if operation.j is not None else None
        i = operation.i - 1

        if operation.op == MatrixOpEnum.A:
            # Finding m_k_i for row k to subtract row k from row i.
            try:
                self.m_dict[(k, i)] = self.output_matrix[k, i] / self.output_matrix[i, i]
            except RuntimeWarning:
                logger.warning("Dividing by zero-close value.")
                self.m_dict[(k, i)] = np.nan

        elif operation.op == MatrixOpEnum.B:
            # Multiplying M_i_j by m_k_i
            try:
                self.n_dict[(k, j, i)] = self.output_matrix[i, j] * self.m_dict[(k, i)]
            except KeyError:
                logger.exception(f"Key {(k, i)} does not exist in {self.m_dict.__name__}")

        elif operation.op == MatrixOpEnum.C:
            # Subtracting n_k_i_j from M_k_j
            try:
                self.output_matrix[k, j] = self.output_matrix[k, j] - self.n_dict[(k, j, i)]
            except KeyError:
                logger.exception(f"Key {(k, j, i)} does not exist in {self.n_dict.__name__}")

        else:
            raise (ValueError(f"No such operation: {str(operation)}"))

    def perform_backward_substitution(self):
        if abs(self.output_matrix[self.n - 1, self.n - 1]) < self.eps:
            logger.error("M[n-1, n-1] = 0 when performing backward substitution.")
            raise (ValueError("M[n-1, n-1] = 0 when performing backward substitution."))

        M = self.output_matrix[:, :self.n]
        y = np.squeeze(self.output_matrix[:, self.n])

        logger.debug(f"M:\n{M}")
        logger.debug(f"y: {y}")

        x = np.zeros_like(y)
        C = np.zeros(self.n)
        x[self.n - 1] = y[self.n - 1] / M[self.n - 1, self.n - 1]

        for i in range(self.n - 2, -1, -1):
            acc = 0.
            for j in range(i + 1, self.n):
                acc += M[i, j] * x[j]
            C[i] = y[i] - acc
            try:
                x[i] = C[i] / M[i, i]
            except RuntimeWarning:
                logger.warning("Dividing by zero-close value.")
                x[i] = np.nan

        self.output_matrix = np.c_[np.eye(self.n), x]
        logger.info(f"Result of backward substitution:\n{self.output_matrix}")

    def write_to_file(self):
        logger.info("Writing to file")

        self.output_str = f"{self.n}\n"
        self.output_str += "\n".join([" ".join(str(x) for x in row) for row in self.output_matrix[:, :self.n]])
        self.output_str += "\n"
        self.output_str += " ".join(str(x) for x in np.squeeze(self.output_matrix[:, self.n]))
        self.output_str += "\n"

        logger.info(f"Output str:\n{self.output_str}")

        with open(self.output_path, "w") as file:
            file.write(self.output_str)

def main():
    input_path = "lin_eq_examples/lin_eq_4x4_input.txt"

    lin_eq_solver = LinearEquationSolver(input_path, "output.txt")
    lin_eq_solver.find_FNF()
    lin_eq_solver.gaussian_elimination_concurrent()
    lin_eq_solver.perform_backward_substitution()
    lin_eq_solver.write_to_file()

if __name__ == "__main__":
    main()
