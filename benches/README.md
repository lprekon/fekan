# Benchmarking
This folder contains a variety of scripts and utilities for benchmarking both fekan on both speed and accuracy

## Speed
### Standard Benchmarks
The [speed_stats.rs](./speed_stats.rs) file can be run with `cargo bench` and will test the speed of [KanLayer's](../src/kan_layer.rs) `forward()`, `backward()`, `update()`, and `update_knots_from_samples()` methods. More information on those methods can be found in the crate documentation

The speed-benchmarking results are stored in [speed_stats.txt](./speed_stats.txt) and are checked in to the repo alongside the tested code

### Flamegraph
the file [flamegraph.sh](./flamegraph.sh) builds and trains a regression model, and generates a flamegraph using [cargo flamegraph](https://github.com/flamegraph-rs/flamegraph). The results are stored in [flamegraph.svg](./flamegraph.svg) and are checked into the repo alongside the tested code.

## Accuracy
While the rust ecosystem comes with excellent support for measure the speed of code, since we're doing machine learning here we also want to measure the capability of our code in addition to the speed, in order to make smart decisions regarding network architecture

The `generate_*_data.py` scripts generate random inputs and the outputs from a particular function, and are used to create datasets for benchmarking. `requirements.txt` has all the dependencies for the python scripts

The `test_*_accuracy.sh` scripts generate a sample dataset using the python scripts with a matching name, then build and train a KAN over the data. The results are stored in the `*.log` files and are checked in to the repo alongside the tested code.

Finally, the included dockerfile can be used to create a docker image with `fekan` and the necessary python dependencies to run the above scripts installed. This is mostly for benchmarking the models on shared-compute resources (e.g AWS Batch or your friend's supercomputer), not intended for any "production" use cases. The image is not currently available on any repositories. To build it, run 
```
docker build -t fekan-benchmark https://github.com/lprekon/fekan -f https://github.com/lprekon/fekan/benches/dockerfile"
```

Example benchmark invocation:
```
docker run fekan-benchmark test_ellipj_accruacy.sh
```