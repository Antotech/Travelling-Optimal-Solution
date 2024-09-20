import numpy as np
import tqdm

from neurons.compare_solutions import compare

if __name__ == '__main__':
    data = [compare(min_node=2000, max_node=5000) for i in tqdm.tqdm(range(3))]
    data = np.array(data)

    print("BASELINE:", data[:, 0].mean())
    print("ANNEALER:", data[:, 1].mean())
    print("MIN.    :", data[:, 2].mean()) 
    print("HPN:", data[:, 3].mean())
    print("CHRIST:", data[:, 4].mean())
    print("ENHANCED:", data[:, 5].mean())
    print("OR_SOLVER:", data[:, 6].mean())
    print("NNS_VALI :", data[:, 7].mean())


