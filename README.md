# fekan
A library to build and train Kolmogorov-Arnold neural networks.

 The `fekan` crate contains utilities to build and train Kolmogorov-Arnold Networks (KANs) in Rust.

 The [kan_layer] module contains the [`kan_layer::KanLayer`] struct, representing a single layer of a KAN,
 which can be used to build full KANs or as a layer in other models.

 The crate also contains the [`Kan`] struct, which represents a full KAN model.

 ## What is a Kolmogorov-Arnold Network?
 Rather than perform a weighted sum of the activations of the previous layer and passing the sum through a fixed non-linear function,
 each node in a KAN passes each activation from the previous layer through a different, trainable non-linear function, then sums and outputs the result.
 This allows the network to be more interpretable than,
 and in some cases be significantly more accurate with a smaller memory footprint than, traditional neural networks.

 Because the activation of each KAN layer can not be calculated using matrix multiplication, training a KAN is currently much slower than training a traditional neural network of comparable size.
 It is the author's hope, however, that the increased accuracy of KANs will allow smaller networks to be used in many cases, offsetting most increased training time;
 and that the interpretability of KANs will more than justify whatever aditional training time remains.

 For more information on the theory behind this library and examples of problem-sets well suited to KANs, see the arXiv paper [KAN: Kolmogorov-Arnold Neural Networks](https://arxiv.org/abs/2404.19756)

 # Examples
 Build, train and save a full KAN regression model with a 2-dimensional input, 1 hidden layer with 3 nodes, and 1 output node,
 where each layer uses degree-4 [B-splines](https://en.wikipedia.org/wiki/B-spline) with 5 coefficients (AKA control points):
 ```rust
 use fekan::kan::{Kan, KanOptions, ModelType};
 use fekan::{Sample, TrainingOptions, EachEpoch};
 use tempfile::tempfile;


 // initialize the model
 let model_options = KanOptions{
     input_size: 2,
     layer_sizes: vec![3, 1],
     degree: 4,
     coef_size: 5,
     model_type: ModelType::Regression,};
 let mut untrained_model = Kan::new(&model_options);

 // train the model
 let training_data: Vec<Sample> = Vec::new();
 /* Load training data */
 # let sample_1 = Sample::new(vec![1.0, 2.0], 3.0);
 # let sample_2 = Sample::new(vec![-1.0, 1.0], 0.0);
 # let training_data = vec![sample_1, sample_2];

 let trained_model = fekan::train_model(untrained_model, &training_data, EachEpoch::DoNotValidateModel, &fekan::EmptyObserver::new(), TrainingOptions::default())?;

 // save the model
 // both Kan and KanLayer implement the serde Serialize trait, so they can be saved to a file using any serde-compatible format
 // here we use the ciborium crate to save the model in the CBOR format
 let mut file = tempfile().unwrap();
 ciborium::into_writer(&trained_model, &mut file)?;
 # Ok::<(), Box<dyn std::error::Error>>(())
 ```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE.txt) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT.txt) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
