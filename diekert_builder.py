import logging
import graphviz

""" Logging setup start """
str_logging_format = "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(str_logging_format)

file_handler = logging.FileHandler("diekert_builder.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
""" Logging setup end """


class DiekertBuilder:
    def __init__(self, matrix_size: int):
        logger.info(f"Created {self.__class__.__name__} instance.")
        self.n = matrix_size
        self.edges = None
        self.digraph = graphviz.Digraph("diekert", filename=f"diekert_graph{matrix_size}")

    def find_edges(self):
        logger.info("Finding edges.")

        self.edges = []

        # E1
        edges1 = []
        for i1 in range(1, self.n):
            for k1 in range(i1 + 1, self.n + 1):
                for j2 in range(i1, self.n + 2):
                    edges1.append((f"A-{k1}-{i1}", f"B-{k1}-{j2}-{i1}"))

        logger.debug(f"edges1 = {edges1}")

        # E2
        edges2 = []
        for i2 in range(2, self.n):
            for k2 in range(i2 + 1, self.n + 1):
                for k1 in (k2, i2):
                    edges2.append((f"C-{k1}-{i2}-{i2-1}", f"A-{k2}-{i2}"))

        logger.debug(f"edges2 = {edges2}")

        # E3
        edges3 = []
        for i1 in range(1, self.n):
            for k1 in range(i1 + 1, self.n + 1):
                for j1 in range(i1, self.n + 2):
                    edges3.append((f"B-{k1}-{j1}-{i1}", f"C-{k1}-{j1}-{i1}"))

        logger.debug(f"edges3 = {edges3}")

        # E4
        edges4 = []
        for i2 in range(2, self.n):
            for k2 in range(i2 + 1, self.n + 1):
                for j2 in range(i2, self.n + 2):
                    if j2 != i2:
                        edges4.append((f"C-{i2}-{j2}-{i2-1}", f"B-{k2}-{j2}-{i2}"))

        logger.debug(f"edges4 = {edges4}")

        # E5
        edges5 = []
        for i1 in range(1, self.n):
            for k1 in range(i1 + 2, self.n + 1):
                for j1 in range(i1 + 1, self.n + 2):
                    if i1 + 1 != j1:
                        edges5.append((f"C-{k1}-{j1}-{i1}", f"C-{k1}-{j1}-{i1+1}"))

        logger.debug(f"edges5 = {edges5}")

        self.edges.extend(edges1)
        self.edges.extend(edges2)
        self.edges.extend(edges3)
        self.edges.extend(edges4)
        self.edges.extend(edges5)

        logger.debug(f"edges = {self.edges}")

    def add_edges(self):
        for edge in self.edges:
            self.digraph.edge(*edge)

    def render_graph(self):
        self.digraph.render(directory="graphs", view=False)

def main():
    logger.info("--- STARTING MAIN ---")

    diekert_builder = DiekertBuilder(matrix_size=3)
    diekert_builder.find_edges()
    diekert_builder.add_edges()
    diekert_builder.render_graph()

    logger.info("--- ENDING MAIN ---")

if __name__ == "__main__":
    main()
