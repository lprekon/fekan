# Branch note
This branch was created to add residual functions to edge output, as described in Liu et. al. However, my implementation is leading to more frequent numerical instability, degrading performance instead of improving it even when the models do converge, and I was able to break through a performance barrier I had hoped this branch would solve simply by choosing better hyperparameters. For all those reasons, I'm not merging this branch in to main. But, I'm going to leave it here in case I want to come back and work on it more later.

# fekan
A library to build and train Kolmogorov-Arnold neural networks.

 The `fekan` crate contains utilities to build and train Kolmogorov-Arnold Networks (KANs) in Rust, including both a struct to represent a full model; and a struct to represant an individual KAN layer, for use on its own or in other models.

 Issues and pull requests are welcome!

 ## What is a Kolmogorov-Arnold Network?
 Rather than perform a weighted sum of the activations of the previous layer and passing the sum through a fixed non-linear function,
 each node in a KAN passes each activation from the previous layer through a different, trainable non-linear function, then sums and outputs the result.
 This allows the network to be more interpretable than,
 and in some cases be significantly more accurate with a smaller memory footprint than, traditional neural networks.

 Because the activation of each KAN layer can not be calculated using matrix multiplication, training a KAN is currently much slower than training a traditional neural network of comparable size.
 It is the author's hope, however, that the increased accuracy of KANs will allow smaller networks to be used in many cases, offsetting most increased training time;
 and that the interpretability of KANs will more than justify whatever aditional training time remains.

 For more information on the theory behind this library and examples of problem-sets well suited to KANs, see the arXiv paper [KAN: Kolmogorov-Arnold Neural Networks](https://arxiv.org/abs/2404.19756)

# Binary usage
The `fekan` crate includes a command-line tool to build and train KANs. Install with 
```sh
cargo install fekan --features serialization
```

Build a new model and train it on a dataset:
```sh
fekan build classifier ...
```
or
```sh
fekan build regressor ...
```

Load an existing model for further training:
```sh
fekan load [model_file] train ...
```

Load an existing model to make predictions on a dataset:
```sh
fekan load [model_file] infer ...
```

Example full command to build a classification model to determine whether a set of features maps to a dog or a cat
```sh
fekan build classifier --data dog_or_cat_data.json --classes "cat,dog" --degree 3 --coefs 4 --hidden-layer-sizes "5,3" --num-epochs 250 --batch-size 100 --knot-adaptivity 0.1 --learning-rate 0.05 --validation-split 0.2 --validate-each-epoch --model-out my_new_model.cbor
```

<details>
<summary>where the data file looks like this...</summary>

```json
[
  {
    features: [1.2, 3.14159, -22.0]
    label: "cat"
  }
  {
    features: [2.89, -0.002, 16.288844]
    label: "dog"
  }
]
```
</details>


For complete usage details use the help command like `fekan help [COMMAND]`

The CLI supports reading data from `.pkl`, `.json`, and ~`.avro`~ files (avro currently bugged). Features must be in a single 'list'-type column/field named "features", and labels (for training) must be in a single column named "labels"; Labels should be strings for classification models and floats for regression models. Models can be saved as pickle, json, or cbor files, and the format is inferred from the provided file extension.

 # Code Example
 Build, train and save a full KAN regression model with a 2-dimensional input, 1 hidden layer with 3 nodes, and 1 output node,
 where each layer uses degree-3 [B-splines](https://en.wikipedia.org/wiki/B-spline) with 10 coefficients (AKA control points):
 ```rust
Build, train and save a full KAN regression model with a 2-dimensional input, 1 hidden layer with 3 nodes, and 1 output node,
where each layer uses degree-4 [B-splines](https://en.wikipedia.org/wiki/B-spline) with 5 coefficients (AKA control points):
```rust
use fekan::kan::{Kan, KanOptions, ModelType};
use fekan::{Sample, training_options::{TrainingOptions, EachEpoch}};
use tempfile::tempfile;



// initialize the model
let model_options = KanOptions{
    input_size: 2,
    layer_sizes: vec![3, 1],
    degree: 3,
    coef_size: 7,
    model_type: ModelType::Regression,
    class_map: None};
let mut untrained_model = Kan::new(&model_options);

// train the model
let training_data: Vec<Sample> = Vec::new();
/* Load training data */
# let sample_1 = Sample::new(vec![1.0, 2.0], 3.0);
# let sample_2 = Sample::new(vec![-1.0, 1.0], 0.0);
# let training_data = vec![sample_1, sample_2];

let trained_model = fekan::train_model(untrained_model, &training_data, TrainingOptions::default())?;

// save the model
// both Kan and KanLayer implement the serde Serialize trait, so they can be saved to a file using any serde-compatible format
// here we use the ciborium crate to save the model in the CBOR format
let mut file = tempfile().unwrap();
ciborium::into_writer(&trained_model, &mut file)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Load and use a trained classification model
 ```rust
 use fekan::kan::Kan;

 let trained_model = serde_json::from_reader(&model_file);
 let data: Vec<Vec<f64>> = /* load data */
 let predictions: Vec<(Vec<f64>, &str)> = Vec::with_capacity(data.len());
 for features in data{
  let logits: Vec<f64> = trained_model.forward(features);
  let (index, probability) = /* interpret your logit data */
  let label: &str = trained_model.node_to_label(index); // get the human-readable label for a given output node
 }
 ```

# To-Do list
`fekan` is fully functional, but there are a number of improvements to make
- Parity with [Liu et. al](https://arxiv.org/abs/2404.19756)
    - [x] grid extension
    - [x] Adjust coefficients on grid update to match previous function
    - [X] pruning un-needed nodes
    - [X] smybolification
    - [ ] visualization
    - [ ] train via methods other than SGD (Adam, LBFGS)
- Speed
    - [x] support multi-threading
    - [ ] support SIMD/parallel computation

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
