import multiprocessing as mp
import os

def foo(queue):
    queue.put([42, None, 'hello'])

def main():
    queue = mp.Queue()
    process = mp.Process(target=foo, args=(queue,))
    process.start()
    print(queue.get())
    process.join()

if __name__ == "__main__":
    main()