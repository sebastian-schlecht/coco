# coco
Training eco-system for Lasagne/Theano neural networks

## Goal
coco allows high-level access to training facilities when training nets with Lasagne or Theano.
Including but not limited to are:
- Easy prefetching of data from disk (file or database)
- One-liner database creation for standard use-cases e.g. classification from folders of images
- On-the-fly realtime data augmention on multiple processor cores using python's multiprocess API.

### Databases
coco allows to easily build datastores from filesystem data. coco can crawl folders for suitable data (images only at the moment) and create databases out of it. The generated databases are compatible to a reader interface to allow data to be read during training, validation or test.

### Augmentation
coco provides a suit of online augmentations to be applied real-time during training. Augmentation is being carried out by the prefetching processes and does thus not block the main training process. Available methods include:
- Random cropping
- Rotation
- Zooming
- Random RGB scaling
- Exposure

### Architectures & Networks
coco provides a variety of networks for easy use. The networks are build upon coco's ```Network``` interface which allows easy load/store functionality as well as pretraining/finetuning facilities.

### Training
TODO
