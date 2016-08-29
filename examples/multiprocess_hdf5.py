import os, sys, inspect
from mince.database_builder import HDF5ClassDatabaseBuilder
from mince.multiprocess import MultiProcessor

"""
Make the lib available here
"""
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

"""
Main program
"""
if __name__ == "__main__":
    # Create a database builder object
    builder = HDF5ClassDatabaseBuilder()
    # Target db location
    db = '/Users/sebastian/Desktop/mince.h5'
    # Build a db from a set of images
    # In case force=false, we do not recreate the db if it's already there!
    builder.build(db, '/Users/sebastian/Desktop/mince_data', shape=(224, 224), force=True)

    # Prepare the reader for read access. This is necessary when combining it with multiprocessors
    builder.setup_read(db)
    # Create a multiprocessor object which manages data loading and transformation daemons
    processor = MultiProcessor(builder, batch_size=2)
    # Start the daemons and tell them to use the databuilder we just setup to pull data from disk
    processor.start_daemons()
    # Iterate over some samples. Note we're using the batch_size from earlier here because the daemons
    # already started doing there thing and we spare some memory op's
    for images, labels in processor.iterate():
        print labels
        print images.mean()
