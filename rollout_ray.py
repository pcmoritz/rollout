import gym
import numpy as np
import ray
import random
import time

@ray.remote
def execute():
    env = gym.make("Pendulum-v0")
    env.reset()
    num_steps = random.randint(0, 100000)
    result = np.zeros((num_steps, 3))
    for i in range(num_steps):
        obs, rew, done, info = env.step([0])
        if done:
            obs = env.reset()
        result[i, :] = obs
    return result

ray.init(num_cpus=2)

ray.get([execute.remote()])

time.sleep(1.0)

t1 = time.time()
ray.get([execute.remote() for i in range(6)])
print("took", time.time() - t1)
