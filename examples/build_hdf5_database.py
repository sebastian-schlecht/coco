import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from mince.database_builder import HDF5ClassDatabaseBuilder

builder = HDF5ClassDatabaseBuilder()

builder.build_read('/Users/sebastian/Desktop/mince.h5', '/Users/sebastian/Desktop/mince_data', shape=(224,224), force=True)

for images, labels in builder.iterate(batch_size=4):
    print labels
    print images.mean()
