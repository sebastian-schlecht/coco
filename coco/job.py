import time, os, errno
import joblib


from os.path import expanduser
home = expanduser("~")
global JOB_DIR
JOB_DIR = "%s/.coco/jobs" % home


class Job(object):
    def __init__(self, name=None):
        if not name:
            name = "job_%s" % str(time.time())
        self.name = name
        self.data = {}

    def save(self, compress=3):
        """
        Persist the job
        :param compress:
        :return:
        """
        filename = "%s/%s" % (JOB_DIR, self.name)
        Job.conditionally_create_dir(filename)
        joblib.dump(self.data, filename, compress=compress)

    def load(self):
        """
        Load the job from disk
        :return:
        """
        filename = "%s/%s" % (JOB_DIR, self.name)
        self.data = joblib.load(filename)

    def set(self, name, value):
        """
        Store a value in the job data dictionary
        :param name:
        :param value:
        :return:
        """
        self.data[name] = value
        self.save()

    @staticmethod
    def list():
        return os.listdir(JOB_DIR)
    
    @staticmethod
    def set_job_dir(dir):
        global JOB_DIR
        JOB_DIR = dir
    @staticmethod
    def from_name(name):
        """
        Create a job from a name as string
        :param name:
        :return:
        """
        j = Job(name)
        j.load()
        return j

    @staticmethod
    def conditionally_create_dir(filename):
        """
        Helper method to conditionally create directories if needed
        :param filename:
        :return:
        """
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
