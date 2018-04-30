import gym
import time
import random
import numpy as np

from mpi4py import MPI

def do_random_rollouts(env):
    num_steps = random.randint(0, 100000)
    result = np.zeros((num_steps, 3))
    for i in range(num_steps):
        obs, rew, done, info = env.step([0])
        if done:
            obs = env.reset()
        result[i, :] = obs
    return result

comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

print(size, rank)

env = gym.make("Pendulum-v0")
env.reset()

t1 = time.time()

rollouts = do_random_rollouts(env)

# time.sleep(random.random())

rootdata = comm.gather(rollouts)
# comm.Barrier()

rollouts = do_random_rollouts(env)

# time.sleep(random.random())

rootdata = comm.gather(rollouts)
# comm.Barrier()

rollouts = do_random_rollouts(env)

# time.sleep(random.random())

rootdata = comm.gather(rollouts)
# comm.Barrier()

if rank == 0:
    print("time was", time.time() - t1)
