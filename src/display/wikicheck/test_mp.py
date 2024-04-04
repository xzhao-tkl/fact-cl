import time
from multiprocessing import Pool

def worker(sl):
    print(sl)
    time.sleep(sl)
    return sl

if __name__ == '__main__':
    with Pool(processes=3) as pool:
        for i in range(5,30,5):
            result = pool.apply_async(func=worker,args=(i,))
        pool.close()
        pool.join()