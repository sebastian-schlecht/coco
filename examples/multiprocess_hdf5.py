import os, sys, inspect

"""
Make the lib available here
"""
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from mince.database_builder import HDF5ClassDatabaseBuilder
from mince.database_reader import HDF5DatabaseReader
from mince.multiprocess import MultiProcessor



"""
Main program
"""
if __name__ == "__main__":
    # Target db location
    db = '/Users/sebastian/Desktop/mince'
    # Build a db from a set of images
    # In case force=false, we do not recreate the db if it's already there!
    files = HDF5ClassDatabaseBuilder.build(db, '/Users/sebastian/Desktop/mince_data_small', shape=(224, 224), force=True)

    reader = HDF5DatabaseReader()
    # Prepare the reader for read access. This is necessary when combining it with multiprocessors
    reader.setup_read(files[0])
    # Create a multiprocessor object which manages data loading and transformation daemons
    processor = MultiProcessor(reader, batch_size=2)
    # Start the daemons and tell them to use the databuilder we just setup to pull data from disk
    processor.start_daemons()
    # Iterate over some samples. Note we're using the batch_size from earlier here because the daemons
    # already started doing there thing and we spare some memory op's
    for images, labels in processor.iterate():
        print labels
        print images.mean()
