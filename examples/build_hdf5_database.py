import os, sys, inspect
from mince.database_builder import HDF5ClassDatabaseBuilder

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Our db path
db = '/Users/sebastian/Desktop/mince.h5'

# New builder object
builder = HDF5ClassDatabaseBuilder()

# Build a db
builder.build(db, '/Users/sebastian/Desktop/mince_data', shape=(224, 224), force=True)

# Setup the reader for read access
builder.setup_read(db)
# And iterate through the data singlethreadedly
for images, labels in builder.iterate(batch_size=4):
    print labels
    print images.mean()
