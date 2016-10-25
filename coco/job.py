import time, os, errno
import joblib

JOB_DIR = "~/.coco/jobs"


class Job(object):
    def __init__(self, name=None):
        if not name:
            name = "job_%s" % str(time.time())
        self.name = name
        self.data = {}

    def save(self):
        filename = "%s/%s" % (JOB_DIR, self.name)
        Job.conditionally_create_dir(filename)
        joblib.dump(self.data, filename)

    def load(self):
        filename = "%s/%s" % (JOB_DIR, self.name)
        joblib.load(filename)

    def set(self, name, value):
        self.data[name] = value
        self.save()

    @staticmethod
    def from_name(name):
        j = Job(name)
        j.load()
        return j

    @staticmethod
    def conditionally_create_dir(filename):
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
