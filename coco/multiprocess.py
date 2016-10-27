from multiprocessing import Process, Queue, Lock
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def thread_proc(queue, func, db, batch_size, lock, wid):
    """
    Main thread procedure
    :param queue: The process-safe queue
    :param func: Callback func for transformation
    :param db: The classdatabase builder instance to call next on
    :param batch_size: The batch size to use
    :param lock: lock to protect the db builder
    :param wid: worker id
    :return:
    """
    if db.randomize_access:
        np.random.seed(42*wid)
        db.permute()
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
        if not self.db.can_read:
            raise AssertionError(
                "Database reader is not setup for read access. Please call setup_read() first on the reader instance.")
        self.q = Queue(maxsize=qsize)
        self.processes = []
        self.batch_size = batch_size
        self.lock = Lock()
        self.daemonized = False
        logger.debug("Creating multiprocessor instance with batchsize=%i and queue_size=%i" % (batch_size, qsize))

    def num_samples(self):
        return self.db.num_samples()
    
    def num_batches(self):
        return self.num_samples() // self.batch_size

    def iterate(self, batches=None):
        """
        Iterate through the dataset by pulling all items out of the queue
        :param batches:
        :return:
        """
        if self.daemonized:
            if batches is None:
                batches = self.db.num_samples() // self.batch_size

            if batches == 0:
                raise UserWarning("Batchsize %i is higher than total number of samples in the dataset %i" % (
                    self.batch_size, self.db.num_samples))

            for _ in range(batches):
                yield self.q.get(block=True)
        else:
            for batch in self.db.iterate(batch_size=self.batch_size, func=self.func):
                yield batch

    def start_daemons(self, parallelism=1):
        """
        Start all daemons
        :param parallelism: int degree of parallelization over various processes
        :return:
        """
        logger.debug("Starting daemons with parallelism=%i", parallelism)
        self.daemonized = True
        if parallelism < 1:
            raise ValueError(
                "Parameter 'parallelism' should be larger than zero. Otherwise no data will be flowing through the pipeline.")
        if parallelism > 1:
            if not self.db.randomize_access:
                logger.warn(
                    "You are spawning multiple database-reader without randomized access order. This will lead to non uniform distributions of data as database-readers cannot share db-cursors at the moment.")

        for wid in range(parallelism):
            args = (self.q, self.func, self.db, self.batch_size, self.lock, wid)
            p = Process(target=thread_proc, args=args)
            p.daemon = True
            self.processes.append(p)
            p.start()
