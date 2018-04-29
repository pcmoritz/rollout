import ray
import random
import time

@ray.remote
def execute():
    time.sleep(random.random())

ray.init(num_cpus=2)

ray.get([execute.remote()])

time.sleep(1.0)

t1 = time.time()
ray.get([execute.remote() for i in range(4)])
print("took", time.time() - t1)
