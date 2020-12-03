import time
from board import get_winning_sets


def get_performance(max_size: int = 10, min_size: int = 3):
    for size in range(min_size, max_size + 1):
        print(f"Testing Speed for size {size}")
        t1 = time.perf_counter()
        winning_sets = get_winning_sets(size, size, size)
        print(winning_sets)
        t2 = time.perf_counter()
        print(t2 - t1)


def build_pareto_frontier:

if __name__ == "__main__":
    get_performance()
