from multiprocessing import Process, Queue, Lock


def thread_proc(queue, func, db, batch_size, lock):
    """
    Main thread procedure
    :param queue: The process-safe queue
    :param func: Callback func for transformation
    :param db: The classdatabase builder instance to call next on
    :param batch_size: The batch size to use
    :param lock: lock to protect the db builder
    :return:
    """
    while True:
        # Pull data from db
        lock.acquire()
        data, labels = db.next_batch(batch_size)
        lock.release()

        # Do processing
        if func is not None:
            data, labels = func(data, labels)
        queue.put((data, labels), block=True)


class MultiProcessor(object):
    def __init__(self, db, func=None, batch_size=8, qsize=20):
        self.func = func
        self.db = db
        self.q = Queue(maxsize=qsize)
        self.processes = []
        self.batch_size = batch_size
        self.lock = Lock()

    def iterate(self):
        """
        Iterate through the dataset by pulling all items out of the queue
        :return:
        """
        num_batches = self.db.num_samples() / self.batch_size
        for _ in range(num_batches):
            yield self.q.get(block=True)

    def start_daemons(self, parallelism=2):
        """
        Start all daemons
        :param parallelism: int degree of parallelization over various processes
        :return:
        """
        for pp in range(parallelism):
            args = (self.q, self.func, self.db, self.batch_size, self.lock)
            p = Process(target=thread_proc, args=args)
            p.daemon = True
            self.processes.append(p)
            p.start()
