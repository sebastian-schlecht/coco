# mince
Training eco-system for Lasagne/Theano neural networks

## Goal
mince allows high-level access to training facilities when training nets with Lasagne or Theano.
Including but not limited to are:
- Easy prefetching of data from disk (file or database)
- One-liner database creation for standard use-cases e.g. classification from folders of images
- On-the-fly realtime data augmention on multiple processor cores using python's multiprocess API
