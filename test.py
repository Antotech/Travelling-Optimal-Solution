import numpy as np
import tqdm

from neurons.compare_solutions import compare

if __name__ == '__main__':
    data = [compare(min_node=2000, max_node=5000) for i in tqdm.tqdm(range(3))]
    data = np.array(data)

    print("BASELINE:", data[:, 0].mean())
    print("ANNEALER:", data[:, 1].mean())
    print("MIN.    :", data[:, 2].mean())



