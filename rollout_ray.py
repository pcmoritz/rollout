import gym
import numpy as np
import ray
import random
import time

NUM_ROLLOUTS_PER_WORKER = 100000

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("num_workers", help="Number of workers to use",
                    type=int)
args = parser.parse_args()

print("args", args)

@ray.remote
def execute():
    env = gym.make("Pendulum-v0")
    env.reset()
    num_steps = random.randint(0, NUM_ROLLOUTS_PER_WORKER)
    result = np.zeros((num_steps, 3))
    for i in range(num_steps):
        obs, rew, done, info = env.step([0])
        if done:
            obs = env.reset()
        result[i, :] = obs
    return result

ray.init(num_cpus=args.num_workers)

ray.get([execute.remote()])

time.sleep(1.0)

t1 = time.time()
ray.get([execute.remote() for i in range(3*args.num_workers)])
print("took", args.num_workers * NUM_ROLLOUTS_PER_WORKER / (time.time() - t1))
