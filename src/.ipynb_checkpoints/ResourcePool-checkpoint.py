import time
from threading import Lock
from concurrent.futures import as_completed, ThreadPoolExecutor, ProcessPoolExecutor



class ResourcePool():
    def __init__(self, resources, thread_num, parallel_type='io-extensive'):
        self.resources = resources
        self.states = [0 for i in range(thread_num)]
        if parallel_type == 'io-extensive':
            self.executor = ThreadPoolExecutor(max_workers=thread_num)
        elif parallel_type == 'cpu-extensive':
            self.executor = ProcessPoolExecutor(max_workers=thread_num)
            
        self.state_lock = Lock()

    def assign(self):
        while True:
            for idx, state in enumerate(self.states):
                if state == 0:
                    with self.state_lock:
                        self.states[idx] = 1
                        # print("Assigned {}, {}".format(idx, self.states))
                    return idx, self.resources[idx]
            time.sleep(0.02)


    def release(self, idx):
        with self.state_lock:
            self.states[idx] = 0
            # print("Released {}, {}".format(idx, self.states))

    def task(self, args):
        res = self.assign()
        print("The {}-th iteration, get resource {}".format(args, res))
        time.sleep(1)
        self.release(res)

    def run(self, iterator):
        futures = []
        for args in iterator:
            future = self.executor.submit(self.task, args)
            futures.append(future)
        
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    thread_num = 8
    resources = [i for i in range(thread_num)]
    rpool = ResourcePool(resources, thread_num)
    rpool.run(range(30))
