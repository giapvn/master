import multiprocessing as mp
import time
import random
import sys

# Using Pool

#
# Functions used by test code
#

def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % (mp.current_process().name, func.__name__, args, result)

def calculatestar(args):
    return calculate(*args)

def mul(a, b):
    time.sleep(0.5 * random.random())
    return a * b

def plus(a, b):
    time.sleep(0.5 * random.random())
    return a + b

def test():
    PROCESSES = 4
    print('Creating pool with %d processes\n' %PROCESSES)

    with mp.Pool(PROCESSES) as pool:
        #
        # Tests
        #

        TASKS = [(mul, (i, 7)) for i in range(10)] + [(plus, (i, 8)) for i in range(10)]
        results = [pool.apply_async(calculate, t) for t in TASKS]
        imap_it = pool.imap(calculatestar, TASKS)

def main():
    mp.freeze_support()
    test()

if __name__ == "__main__":
    main()