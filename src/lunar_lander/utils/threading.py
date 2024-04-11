import os
import time
from multiprocessing.pool import Pool


class MultiCoreController:
    """Simple multiprocessing helper to fan out parameter sweeps."""

    def __init__(self, n_cores: int) -> None:
        cpu_count = os.cpu_count() or 1
        if n_cores > cpu_count:
            raise Exception(
                "It's impossible to utilize more instances than available CPU cores!")

        self.pool = Pool(processes=n_cores)

    def run(self, function: (...), param_list: list[tuple[any]]) -> None:
        results = []

        for params in param_list:
            results.append(self.pool.apply_async(func=function, args=params))
            print(
                f"Started a new instance for function '{function.__name__}'.")
            time.sleep(5)

        self.pool.close()
        self.pool.join()
        final_results = [result.get() for result in results]
