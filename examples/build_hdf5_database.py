import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from mince.database_builder import HDF5ClassDatabaseBuilder
from mince.database_reader import HDF5DatabaseReader

"""
Main program
"""
if __name__ == "__main__":
    # Our db path
    db = '/Users/sebastian/Desktop/mince'
    # Build a db
    files = HDF5ClassDatabaseBuilder.build(db, '/Users/sebastian/Desktop/mince_data_small', shape=(224, 224), force=True)

    print files
    reader = HDF5DatabaseReader()
    reader.setup_read(files[0], True)
    for batch in reader.iterate(batch_size=8):
        input, label = batch
        print input.mean()
