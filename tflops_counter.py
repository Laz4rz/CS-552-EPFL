# for some reason this script does not terminate immediately after the last print statement
# unless synchronize() is called
# python -m trace --trace tflops_counter.py>logSynchronize.txt

import time
import timeit
import logging

import torch
import numpy as np

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s\n')

# def test_tflops(size, runs, device):
#     runtimes = []
#     for i in range(runs):
#         m1 = torch.randn(size, size, device=device, requires_grad=False)

#         start = time.time()

#         m1 = m1 @ m1

#         runtime = time.time()-start
#         runtimes.append(runtime*1000)

#     runtimes = np.array(runtimes)
#     logging.info(f"Device: {device}, Size: {size}, Runs: {runs}")
#     logging.info(f"{(np.mean(runtimes))*1000:.5f}ms")
#     logging.info(f"TFLOPS: {2*size*size*1000/(np.mean(runtimes)*1e12):.5f}+-{2*size*size*1000/(np.std(runtimes)*1e12):.5f}")


# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     synchronize = torch.mps.synchronize
# else:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     synchronize = torch.cuda.synchronize

# test_tflops(1024, 100, torch.device("cpu"))
# test_tflops(1024, 100, torch.device("mps"))

# # synchronize()
# logging.info("Done")

def test_cpu():
    a_cpu = torch.rand(1000, device='cpu')
    b_cpu = torch.rand((1000, 1000), device='cpu')
    a_cpu @ b_cpu
def test_mps():
    a_mps = torch.rand(1000, device='mps')
    b_mps = torch.rand((1000, 1000), device='mps')
    a_mps @ b_mps

print('cpu', timeit.timeit(lambda: test_cpu(), number=1000))
print('mps', timeit.timeit(lambda: test_mps(), number=1000))
