import time
import random

from mpi4py import MPI

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

print(size, rank)

t1 = time.time()

time.sleep(random.random())

comm.Barrier()

time.sleep(random.random())

comm.Barrier()

if rank == 0:
    print("time was", time.time() - t1)
