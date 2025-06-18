This codebase is built upon the key vector.py file, which defines functions for creating and performing key operations with hyperdimensional binary vectors (hypervectors). 

The goal of this project is to create recreate a classic image classification model, but built with solely hypervectors (vectors of length ~10,000) instead of a multilayer CNN.

By harnessing the unique properties of randomly generated vectors within a high dimensional space, it was possible to create meaningful representational relationships that enable supervised learning purely with binary vectors. 

A large portion of this project is exploring different encoding methods for image data, which are explored in `/hil_torch/modalities`. The simplest approach was encoding via spatial intestity, but other, more interesting approaches (such as encoding via histogram or the encoding of convolutional features), are also explored.

The intuition behind this codebase is large in thanks to Pentti Kanerva's [foundational HDC paper](https://rctn.org/vs265/kanerva09-hyperdimensional.pdf) and Peter Sutor's [HD Glue paper](https://arxiv.org/pdf/2205.15534).


