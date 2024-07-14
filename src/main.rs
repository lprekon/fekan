// use std::fs::File;

//! Does this show up anywhere
use std::{error::Error, fs::File, path::PathBuf};

use clap::{ArgGroup, Args, Parser, Subcommand};

use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model,
    training_observer::TrainingObserver,
    validate_model, EachEpoch, Sample, TrainingOptions,
};
use indicatif::{ProgressBar, ProgressStyle};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;

/// A simple CLI for training and using Kolmogorov-Arnold neural networks. Appropriate for datasets that can be loaded into memory.
#[derive(Parser, Debug, Clone)]
struct Cli {
    #[command(subcommand)]
    command: WhereCommands,

    /// log model output to stdout in addition to drawing on the terminal, allowing output to be piped
    #[arg(long, default_value = "false", global = true)]
    log_output: bool,
}

#[derive(Subcommand, Debug, Clone)]
enum WhereCommands {
    /// Build a new model and train it on the provided data
    Build(BuildArgs),
    /// Load an existing model for either further training or for inference
    Load(LoadArgs),
}

#[derive(Args, Clone, Debug)]
struct BuildArgs {
    /// Classification or regression model. If classifier, ||output_layer|| = ||classes||, if regression, ||output_layer|| = 1
    #[command(subcommand)]
    model_type: CliModelType,
}

#[derive(Subcommand, Debug, Clone)]
enum CliModelType {
    // ! this enum should exactly match the fekan::ModelType enum
    // ! This is a workaround for the fact that the ValueEnum derive macro doesn't support enums with associated data, so we can't wrap the ModelType enum in a new enum
    /// Build a model to do classification with the provided classes
    Classifier(ClassifierArgs),
    /// Build a regression model to predict one or more continuous output values
    Regressor(RegressorArgs),
}

#[derive(Args, Clone, Debug)]
struct ClassifierArgs {
    /// A comma-separated list of classes which the model will be trained to predict. The model will have ||classes|| output nodes.
    /// Any data points with labels not in this list will be ignored
    #[arg(short, long, required = true, value_delimiter = ',')]
    classes: Vec<String>,

    #[command(flatten)]
    params: GenericBuildParams,
}

#[derive(Args, Clone, Debug)]
struct RegressorArgs {
    // #[arg(long = "output-nodes", default_value = "1")]
    /// The number of output nodes in the model. If > 1, the model will be a multi-output regression model
    // num_output_nodes: usize, not currently implemented

    /// a comma-separated list of human-readable labels for the output nodes of the model.
    /// the model will have ||labels|| output nodes, or 1 output node if no labels are provided
    #[arg(long, value_delimiter = ',')]
    labels: Option<Vec<String>>,

    #[command(flatten)]
    params: GenericBuildParams,
}

#[derive(Args, Clone, Debug)]
struct GenericBuildParams {
    #[arg(short = 'k', long, default_value = "3", global = true)]
    /// The degree of the spline basis functions
    degree: usize,

    #[arg(long = "coefs", default_value = "4", global = true)]
    /// The number of coefficients/control points in the spline basis functions
    num_coefficients: usize,

    #[arg(long, alias = "hl", global = true, value_delimiter = ',')]
    /// a comma-separated list of hidden layer sizes. If empty, the model will only have the output layer
    hidden_layer_sizes: Option<Vec<usize>>,
    // #[command(flatten)]
    #[command(flatten)]
    training_parameters: TrainArgs,
}

#[derive(Args, Clone, Debug)]
#[command(group(ArgGroup::new("output").required(true).multiple(false)))]
struct TrainArgs {
    /// path to the file containing the training data.
    /// The file format is determined by the file extension. Supported formats are: pickle, json, avro
    /// Features should be in an ordered list in single column/field named 'features', and labels should be in a single column/field named 'label'
    #[arg(short = 'd', long = "data")]
    data_file: PathBuf,

    #[arg(
        short = 'e',
        long,
        visible_alias = "epochs",
        default_value = "100",
        global = true
    )]
    /// number of epochs to train the model for
    num_epochs: usize,

    #[arg(long, default_value = "100", global = true)]
    /// the interval at which to update the spline knot vectors, based on the inputs seen since the last update
    knot_update_interval: usize,

    #[arg(long, default_value = "0.1", global = true)]
    /// A hyperparameter used when updating knots from recent samples.
    /// When knot_adaptivity = 0, the knots are uniformly distributed over the sample space;
    /// when knot_adaptivity = 1, the knots are placed at the quantiles of the input data;
    /// 0 < knot_adaptivity < 1 interpolates between these two extremes.
    knot_adaptivity: f64,

    #[arg(long, alias = "lr", default_value = "0.01", global = true)]
    /// the learning rate used to update the model weights
    learning_rate: f64,

    /// the maximum length to which knot vectors can be extended during training. If not set, the knot vectors will gradually be extended so that the total number of knots ~= the number of training samples
    #[arg(long, global = true)]
    max_knot_length: Option<usize>,

    #[arg(long, global = true)]
    /// if set, the model will be run against the validation data after each epoch, and the loss will be reported to the observer
    validate_each_epoch: bool,

    #[arg(long, default_value = "0.2", global = true)]
    /// the fraction of the training data to use as validation data
    validation_split: f64,

    /// path to the output file for the model weights. Supported file extensions are .pkl, .json, and .cbor
    #[arg(short = 'o', long = "model-out", group = "output")]
    model_output_file: Option<PathBuf>,

    /// if set, the model will not be saved to a file after training. Useful for experimentation, as saving the model can be slow
    #[arg(long, group = "output")]
    no_save: bool,
}

#[derive(Args, Clone, Debug)]
struct LoadArgs {
    /// path to the model weights file.
    model_input_file: PathBuf,

    /// A list of comma-separated human-readable labels for the output nodes of the model. Required for training classification models, not used for regression models.
    /// NOTE: make sure the order of the labels matches the order of the classes used to train the model and that none are missing
    #[arg(long, value_delimiter = ',', global = true)]
    classes: Option<Vec<String>>,

    #[command(subcommand)]
    command: WhyCommands,
}

#[derive(Subcommand, Debug, Clone)]
enum WhyCommands {
    /// Further train the model weights loaded from a file
    Train(TrainArgs),
    /// Use a loaded model to make predictions on new data
    Infer {
        /// path to the file containing the data. currently only supports pickle files
        #[arg(short = 'd', long = "data", required = true)]
        data_file: PathBuf, // needed because the data file is stored in train args everywhere else
    },
}

// when designing this program, I had 3 things for which I could optimize - a helpful and intutive CLI,
// consistent and deduplicated definitions for the arguments, and a clean main function - and I could pick two.
// Please forgive the messy code below :)
fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    println!("Using arguments {cli:?}");
    match cli.command {
        WhereCommands::Build(build_args) => match build_args.model_type {
            CliModelType::Classifier(classifier_args) => {
                let train_args = classifier_args.params.training_parameters;

                // check the output file extension to make sure we can save it later. If not, better to fail now than after training
                if let Some(output_file_path) = &train_args.model_output_file {
                    validate_output_file_extension(output_file_path);
                }

                let (training_data, validation_data) = load_classification_data(
                    &train_args.data_file,
                    train_args.validation_split,
                    &classifier_args.classes,
                )?;

                let (passed_validation_data, observer_ticks) = determine_validation_data_and_ticks(
                    &train_args,
                    &validation_data,
                    &training_data,
                );
                let training_observer =
                    TrainingProgress::new(observer_ticks as u64, cli.log_output);

                // build our list of layer sizes, which should equal all the hidden layers specified by the user, plus the output layer
                // `layers` only needs to be mutable while we build it, then should be immutable after that. Using a closure to build it accomplishes this nicely
                let layers = {
                    let mut layers = classifier_args
                        .params
                        .hidden_layer_sizes
                        .unwrap_or_default();
                    layers.push(classifier_args.classes.len() as usize);
                    layers
                };
                // build the model
                let untrained_model = Kan::new(&KanOptions {
                    input_size: training_data[0].features().len() as usize,
                    layer_sizes: layers,
                    degree: classifier_args.params.degree,
                    coef_size: classifier_args.params.num_coefficients,
                    model_type: ModelType::Classification,
                    class_map: Some(classifier_args.classes),
                });

                let training_options = TrainingOptions {
                    // we can't use the From<T> pattern here because some of the fields are not directly copyable
                    num_epochs: train_args.num_epochs,
                    knot_update_interval: train_args.knot_update_interval,
                    knot_adaptivity: train_args.knot_adaptivity,
                    learning_rate: train_args.learning_rate,
                    max_knot_length: train_args.max_knot_length,
                };

                // run the training loop on the model
                let mut trained_model = train_model(
                    untrained_model,
                    &training_data,
                    passed_validation_data,
                    &training_observer,
                    training_options,
                )?;
                if !train_args.validate_each_epoch {
                    // we didn't validate each epoch, so we need to validate the model now
                    let validation_loss =
                        validate_model(&validation_data, &mut trained_model, &training_observer);
                    training_observer.into_inner().println(format!(
                        "{} Final validation Loss: {}",
                        chrono::Local::now(),
                        validation_loss
                    ));
                } else {
                    training_observer
                        .into_inner()
                        .finish_with_message("Training complete");
                }
                // save the model to a file
                if let Some(model_output_file) = &train_args.model_output_file {
                    serialize_model(model_output_file, &trained_model)?;
                }
                Ok(())
            }
            CliModelType::Regressor(regressor_args) => {
                let train_args = regressor_args.params.training_parameters;

                // check the output file extension to make sure we can save it later. If not, better to fail now than after training
                if let Some(output_file_path) = &train_args.model_output_file {
                    validate_output_file_extension(output_file_path);
                }

                let (training_data, validation_data) =
                    load_regression_data(&train_args.data_file, train_args.validation_split)?;

                let (passed_validation_data, observer_ticks) = determine_validation_data_and_ticks(
                    &train_args,
                    &validation_data,
                    &training_data,
                );
                let training_observer =
                    TrainingProgress::new(observer_ticks as u64, cli.log_output);

                // build our list of layer sizes, which should equal all the hidden layers specified by the user, plus the output layer
                let layers = {
                    let mut layers = regressor_args.params.hidden_layer_sizes.unwrap_or_default();
                    let output_size = regressor_args
                        .labels
                        .as_ref()
                        .map_or(1, |labels| labels.len());
                    layers.push(output_size);
                    layers
                };
                // build the model
                let untrained_model = Kan::new(&KanOptions {
                    input_size: training_data[0].features().len() as usize,
                    layer_sizes: layers,
                    degree: regressor_args.params.degree,
                    coef_size: regressor_args.params.num_coefficients,
                    model_type: ModelType::Regression,
                    class_map: regressor_args.labels,
                });

                let training_options = TrainingOptions {
                    // we can't use the From<T> pattern here because some of the fields are not directly copyable
                    num_epochs: train_args.num_epochs,
                    knot_update_interval: train_args.knot_update_interval,
                    knot_adaptivity: train_args.knot_adaptivity,
                    learning_rate: train_args.learning_rate,
                    max_knot_length: train_args.max_knot_length,
                };

                // run the training loop on the model
                let mut trained_model = train_model(
                    untrained_model,
                    &training_data,
                    passed_validation_data,
                    &training_observer,
                    training_options,
                )?;

                if !train_args.validate_each_epoch {
                    // we didn't validate each epoch, so we need to validate the model now
                    let validation_loss =
                        validate_model(&validation_data, &mut trained_model, &training_observer);
                    training_observer.into_inner().println(format!(
                        "{} Final validation Loss: {}",
                        chrono::Local::now(),
                        validation_loss
                    ));
                } else {
                    training_observer
                        .into_inner()
                        .finish_with_message("Training complete");
                }

                if let Some(model_output_file) = &train_args.model_output_file {
                    serialize_model(model_output_file, &trained_model)?;
                }
                Ok(())
            }
        },
        WhereCommands::Load(load_args) => {
            let mut loaded_model = deserialize_model(&load_args.model_input_file)?;

            match load_args.command {
                WhyCommands::Train(train_args) => {
                    let training_options = TrainingOptions {
                        // we can't use the From<T> pattern here because some of the fields are not directly copyable
                        num_epochs: train_args.num_epochs,
                        knot_update_interval: train_args.knot_update_interval,
                        knot_adaptivity: train_args.knot_adaptivity,
                        learning_rate: train_args.learning_rate,
                        max_knot_length: train_args.max_knot_length,
                    };

                    let load_result = match loaded_model.model_type() {
                        ModelType::Classification => load_classification_data(
                            &train_args.data_file,
                            train_args.validation_split,
                            loaded_model
                                .class_map()
                                .expect("Classification model has no class map"),
                        ),

                        ModelType::Regression => {
                            load_regression_data(&train_args.data_file, train_args.validation_split)
                        }
                    };
                    let (training_data, validation_data) = load_result?;

                    // if the user wants the model validated each epoch, pass the validation data to the training function and included the counts in the training observer.
                    let (passed_validation_data, observer_ticks) =
                        determine_validation_data_and_ticks(
                            &train_args,
                            &validation_data,
                            &training_data,
                        );
                    let training_observer =
                        TrainingProgress::new(observer_ticks as u64, cli.log_output);

                    // run the training loop on the model
                    let mut trained_model = train_model(
                        loaded_model,
                        &training_data,
                        passed_validation_data,
                        &training_observer,
                        training_options,
                    )?;
                    if !train_args.validate_each_epoch {
                        // we didn't validate each epoch, so we need to validate the model now
                        let validation_loss = validate_model(
                            &validation_data,
                            &mut trained_model,
                            &training_observer,
                        );
                        training_observer.into_inner().println(format!(
                            "{} Final validation Loss: {}",
                            chrono::Local::now(),
                            validation_loss
                        ));
                    } else {
                        training_observer
                            .into_inner()
                            .finish_with_message("Training complete");
                    }
                    // save the model to a file
                    if let Some(model_output_file) = &train_args.model_output_file {
                        serialize_model(model_output_file, &trained_model)?;
                    }
                    Ok(())
                }
                WhyCommands::Infer { data_file } => {
                    let data = load_inference_data(&data_file)?;
                    for sample in data.iter() {
                        let activation = loaded_model.forward(sample.features().clone())?;
                        let output_string = "[".to_string()
                            + &activation
                                .iter()
                                .enumerate()
                                .map(|(idx, a)| {
                                    format!(
                                        "{}: {}",
                                        loaded_model
                                            .node_to_label(idx)
                                            .unwrap_or("[IDX OUT OF BOUNDS]"),
                                        a
                                    )
                                })
                                .collect::<Vec<String>>()
                                .join(", ")
                            + "]";
                        println!("{}", output_string);
                    }
                    Ok(())
                }
            }
        }
    }
}

fn determine_validation_data_and_ticks<'a>(
    train_args: &TrainArgs,
    validation_data: &'a [Sample],
    training_data: &[Sample],
) -> (EachEpoch<'a>, usize) {
    let (passed_validation_data, observer_ticks) = if train_args.validate_each_epoch {
        (
            EachEpoch::ValidateModel(&validation_data),
            (training_data.len() + validation_data.len()) * train_args.num_epochs,
        )
    } else {
        (
            EachEpoch::DoNotValidateModel,
            training_data.len() * train_args.num_epochs + validation_data.len(),
        )
    };
    (passed_validation_data, observer_ticks)
}

fn serialize_model(
    model_output_file: &PathBuf,
    trained_model: &Kan,
) -> Result<File, Box<dyn Error>> {
    println!("Saving model to file: {:?}", model_output_file);
    let mut out_file = File::create(model_output_file)?;
    let file_extension = model_output_file
        .extension()
        .expect("No file extension found for output file - unable to determine output format")
        .to_str()
        .expect("Error converting file extension to string");
    match file_extension {
        "pkl" => serde_pickle::to_writer(&mut out_file, trained_model, Default::default())?,
        "json" => serde_json::to_writer(&mut out_file, trained_model)?,
        "cbor" => ciborium::into_writer(trained_model, &mut out_file)?,
        _ => panic!("Unsupported file extension: {}", file_extension),
    }
    Ok(out_file)
}

fn validate_output_file_extension(output_file_path: &PathBuf) {
    let file_extension = output_file_path
        .extension()
        .expect("No file extension found for output file - unable to determine output format")
        .to_str()
        .expect("Error converting file extension to string");
    match file_extension {
        "pkl" | "json" | "cbor" => (),
        _ => panic!("Unsupported file extension: {}", file_extension),
    }
}

/// panic if the file extension is not supported or the model can't be loaded
fn deserialize_model(model_input_file: &PathBuf) -> Result<Kan, Box<dyn Error>> {
    println!("Loading model from file: {:?}", model_input_file);
    let file = File::open(model_input_file)?;
    let file_extension = model_input_file
        .extension()
        .expect("No file extension found for input file - unable to determine input format")
        .to_str()
        .expect("Error converting file extension to string");
    let model: Kan = match file_extension {
        "pkl" => serde_pickle::from_reader(file, Default::default())?,
        "json" => serde_json::from_reader(file)?,
        "cbor" => ciborium::from_reader(file)?,
        _ => panic!("Unsupported file extension"),
    };
    Ok(model)
}

struct TrainingProgress {
    pb: ProgressBar,
    should_log: bool,
}

impl TrainingProgress {
    fn new(total: u64, should_log: bool) -> Self {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] [{bar:40.green/white}] {human_pos}/{human_len} {per_sec} ({eta}) {msg}",
                )
                .unwrap(),
        );
        TrainingProgress { pb, should_log }
    }

    fn into_inner(self) -> ProgressBar {
        self.pb
    }
}

impl TrainingObserver for TrainingProgress {
    fn on_sample_end(&self) {
        self.pb.inc(1);
    }
    fn on_epoch_end(&self, epoch: usize, epoch_loss: f64, validation_loss: f64) {
        self.pb.println(format!(
            "{} Epoch {}: Training Loss: {}, Validation Loss: {}",
            chrono::Local::now(),
            epoch,
            epoch_loss,
            validation_loss
        ));
        if self.should_log {
            println!(
                "{} Epoch {}: Training Loss: {}, Validation Loss: {}",
                chrono::Local::now(),
                epoch,
                epoch_loss,
                validation_loss
            );
        }
    }
}

#[derive(Deserialize, Debug)]
#[cfg_attr(test, derive(serde::Serialize, PartialEq))]
struct ClassificationSample {
    features: Vec<f64>,
    label: String,
}

#[derive(Deserialize, Debug)]
#[cfg_attr(test, derive(serde::Serialize, PartialEq))]
struct RegressionSample {
    features: Vec<f64>,
    label: f64,
}

const SUPPORTED_EXTENSIONS: [&str; 3] = ["pkl", "json", "avro"];

fn load_classification_data(
    data_file_path: &PathBuf,
    validation_split: f64,
    classes: &Vec<String>,
) -> Result<(Vec<Sample>, Vec<Sample>), Box<dyn Error>> {
    let file = File::open(data_file_path)?;
    let file_extension = data_file_path
        .extension()
        .expect("UNABLE TO LOAD DATA: no file extension found")
        .to_str()
        .expect("UNABLE TO LOAD DATA: unable to convert file extension to string");
    // load the raw data with string labels
    let raw_data: Vec<ClassificationSample> = match file_extension {
        "pkl" => serde_pickle::from_reader(file, Default::default())?,
        "json" => serde_json::from_reader(file)?,
        "avro" => {
            let mut data: Vec<ClassificationSample> = vec![];
            let avro_reader = apache_avro::Reader::new(file)?;
            for value in avro_reader {
                data.push(apache_avro::from_value(&value.expect("Bad avro value"))?);
            }
            data
        }

        _ => panic!(
            "UNABLE TO LOAD DATA: unsupported file extension: {}. Supported extensions are: {}",
            file_extension,
            SUPPORTED_EXTENSIONS.join(", ")
        ),
    };
    let rows_loaded = raw_data.len();

    // parse the data, maping the label string to a u32
    let class_map: FxHashMap<String, u32> = FxHashMap::from_iter(
        classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i as u32)),
    );
    println!("Using class map: {:?}", class_map);

    // turn the raw data into a vector of Samples (with the labels mapped to u32s (with the u32s  as f64s))
    let mut data = Vec::with_capacity(raw_data.len());
    for raw_sample in raw_data {
        if !class_map.contains_key(&raw_sample.label) {
            continue; // ignore data points with labels not in the provided class list
        }
        let label = class_map[&raw_sample.label];
        let features: Vec<f64> = raw_sample.features;
        data.push(Sample::new(features, label as f64));
    }

    split_data(validation_split, data, rows_loaded)
}

fn load_regression_data(
    data_file_path: &PathBuf,
    validation_split: f64,
) -> Result<(Vec<Sample>, Vec<Sample>), Box<dyn Error>> {
    println!("Loading regression data from file: {:?}", data_file_path);
    let file = File::open(data_file_path)?;
    let file_extension = data_file_path
        .extension()
        .expect("UNABLE TO LOAD DATA: no file extension found")
        .to_str()
        .expect("UNABLE TO LOAD DATA: unable to convert file extension to string");
    let data: Vec<RegressionSample> = match file_extension {
        "pkl" => serde_pickle::from_reader(file, Default::default())?,
        "json" => serde_json::from_reader(file)?,
        "avro" => {
            let mut data: Vec<RegressionSample> = vec![];
            let avro_reader = apache_avro::Reader::new(file)?;
            for value in avro_reader {
                data.push(apache_avro::from_value(&value.expect("Bad avro value"))?);
            }
            data
        }
        _ => panic!(
            "UNABLE TO LOAD DATA: unsupported file extension: {}. Supported extensions are: {}",
            file_extension,
            SUPPORTED_EXTENSIONS.join(", ")
        ),
    };
    let rows_loaded = data.len();

    // turn the raw data into a vector of Samples
    let data: Vec<Sample> = data
        .into_iter()
        .map(|raw_sample| Sample::new(raw_sample.features, raw_sample.label))
        .collect();

    println!("creating validation set");
    // separate the data into training and validation sets
    split_data(validation_split, data, rows_loaded)
}

fn load_inference_data(data_file_path: &PathBuf) -> Result<Vec<Sample>, Box<dyn Error>> {
    #[derive(Deserialize, Debug)]
    struct InferenceSample {
        features: Vec<f64>,
    }

    let file = File::open(data_file_path)?;
    let file_extension = data_file_path
        .extension()
        .expect("UNABLE TO LOAD DATA: no file extension found")
        .to_str()
        .expect("UNABLE TO LOAD DATA: unable to convert file extension to string");
    let raw_data: Vec<InferenceSample> = match file_extension {
        "pkl" => serde_pickle::from_reader(file, Default::default())?,
        "json" => serde_json::from_reader(file)?,
        "avro" => {
            let mut data: Vec<InferenceSample> = vec![];
            let avro_reader = apache_avro::Reader::new(file)?;
            for value in avro_reader {
                data.push(apache_avro::from_value(&value.expect("Bad avro value"))?);
            }
            data
        }
        _ => panic!(
            "UNABLE TO LOAD DATA: unsupported file extension: {}. Supported extensions are: {}",
            file_extension,
            SUPPORTED_EXTENSIONS.join(", ")
        ),
    };

    // turn the raw data into a vector of Samples
    Ok(raw_data
        .into_iter()
        .map(|raw_sample| Sample::new(raw_sample.features, 0.0))
        .collect())
}

fn split_data(
    validation_split: f64,
    data: Vec<Sample>,
    rows_loaded: usize,
) -> Result<(Vec<Sample>, Vec<Sample>), Box<dyn Error>> {
    // separate the data into training and validation sets
    let mut validation_indecies: FxHashSet<usize> = FxHashSet::default();
    while validation_indecies.len() < (validation_split * data.len() as f64) as usize {
        let index = rand::random::<usize>() % data.len();
        validation_indecies.insert(index);
    }
    let mut training_data: Vec<Sample> = Vec::with_capacity(data.len() - validation_indecies.len());
    let mut validation_data: Vec<Sample> = Vec::with_capacity(validation_indecies.len());
    for (i, sample) in data.into_iter().enumerate() {
        if validation_indecies.contains(&i) {
            validation_data.push(sample);
        } else {
            training_data.push(sample);
        }
    }
    assert!(
        training_data.len() + validation_data.len() == rows_loaded,
        "Data split error. Training: {}, Validation: {}, Total: {}",
        training_data.len(),
        validation_data.len(),
        rows_loaded
    );
    println!(
        "Data loaded. Training: {}, Validation: {}",
        training_data.len(),
        validation_data.len()
    );
    Ok((training_data, validation_data))
}

#[cfg(test)]
mod test_main {
    use super::*;

    use std::{io::Seek, vec};
    use tempfile::tempdir;

    use crate::{ClassificationSample, RegressionSample};

    fn classification_data() -> (Vec<ClassificationSample>, Vec<String>) {
        let classes = vec![
            "a".to_string(),
            "0123456789".to_string(),
            "!@#$%^&*()~<>?:\"{}\\|'éñ漢字".to_string(),
        ];
        (
            vec![
                ClassificationSample {
                    features: vec![1.0, 2.0, 3.0],
                    label: classes[0].clone(),
                },
                ClassificationSample {
                    features: vec![f64::MIN, 0.0, f64::MAX],
                    label: classes[1].clone(),
                },
                ClassificationSample {
                    features: vec![-1.1, -2.5, 9.7],
                    label: classes[2].clone(),
                },
            ],
            classes,
        )
    }

    fn regression_data() -> Vec<RegressionSample> {
        vec![
            RegressionSample {
                features: vec![1.0, 2.0, 3.0],
                label: -1.0,
            },
            RegressionSample {
                features: vec![f64::MIN, 0.0, f64::MAX],
                label: 0.001,
            },
            RegressionSample {
                features: vec![-1.1, -2.5, 9.7],
                label: 3.14,
            },
        ]
    }

    #[test]
    fn read_pickle_classification_data() {
        let tmp_dir = tempdir().unwrap();
        let file_path = tmp_dir.path().join("test.pkl");
        let mut file = File::create(&file_path).unwrap();
        let (test_data, class_list) = classification_data();
        serde_pickle::to_writer(&mut file, &test_data, Default::default()).unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let (loaded_data, _) = load_classification_data(&file_path, 0.0, &class_list).unwrap();
        let expected_data: Vec<Sample> = test_data
            .iter()
            .enumerate()
            .map(|(i, sample)| Sample::new(sample.features.clone(), i as f64))
            .collect();
        assert_eq!(expected_data, loaded_data);
    }

    #[test]
    fn read_json_classifcation_data() {
        let tmp_dir = tempdir().unwrap();
        let file_path = tmp_dir.path().join("test.json");
        let mut file = File::create(&file_path).unwrap();
        let (test_data, class_list) = classification_data();
        serde_json::to_writer(&mut file, &test_data).unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let (loaded_data, _) = load_classification_data(&file_path, 0.0, &class_list).unwrap();
        let expected_data: Vec<Sample> = test_data
            .iter()
            .enumerate()
            .map(|(i, sample)| Sample::new(sample.features.clone(), i as f64))
            .collect();
        assert_eq!(expected_data, loaded_data);
    }

    #[test]
    fn read_avro_classifcation_data() {
        let tmp_dir = tempdir().unwrap();
        let file_path = tmp_dir.path().join("test.avro");
        let mut file = File::create(&file_path).unwrap();
        let (test_data, class_list) = classification_data();
        let schema = apache_avro::Schema::parse_str(r#"{"type": "record", "name": "test", "fields": [{"name": "features", "type": {"type": "array", "items": "float"}}, {"name": "label", "type": "string"}]}"#).unwrap();
        let mut writer = apache_avro::Writer::new(&schema, &mut file);
        for sample in &test_data {
            writer
                .append(apache_avro::to_value(sample).unwrap())
                .unwrap();
        }
        writer.flush().unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let (loaded_data, _) = load_classification_data(&file_path, 0.0, &class_list).unwrap();
        let expected_data: Vec<Sample> = test_data
            .iter()
            .enumerate()
            .map(|(i, sample)| Sample::new(sample.features.clone(), i as f64))
            .collect();
        assert_eq!(expected_data, loaded_data);
    }

    #[test]
    fn read_pickle_regression_data() {
        let tmp_dir = tempdir().unwrap();
        let file_path = tmp_dir.path().join("test.pkl");
        let mut file = File::create(&file_path).unwrap();
        let test_data = regression_data();
        serde_pickle::to_writer(&mut file, &test_data, Default::default()).unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let (loaded_data, _) = load_regression_data(&file_path, 0.0).unwrap();
        let expected_data: Vec<Sample> = test_data
            .iter()
            .map(|sample| Sample::new(sample.features.clone(), sample.label))
            .collect();
        assert_eq!(expected_data, loaded_data);
    }

    #[test]
    fn read_json_regression_data() {
        let tmp_dir = tempdir().unwrap();
        let file_path = tmp_dir.path().join("test.json");
        let mut file = File::create(&file_path).unwrap();
        let test_data = regression_data();
        serde_json::to_writer(&mut file, &test_data).unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let (loaded_data, _) = load_regression_data(&file_path, 0.0).unwrap();
        let expected_data: Vec<Sample> = test_data
            .iter()
            .map(|sample| Sample::new(sample.features.clone(), sample.label))
            .collect();
        assert_eq!(expected_data, loaded_data);
    }

    #[test]
    fn read_avro_regression_data() {
        let tmp_dir = tempdir().unwrap();
        let file_path = tmp_dir.path().join("test.avro");
        let mut file = File::create(&file_path).unwrap();
        let test_data = regression_data();
        let schema = apache_avro::Schema::parse_str(r#"{"type": "record", "name": "test", "fields": [{"name": "features", "type": {"type": "array", "items": "float"}}, {"name": "label", "type": "float"}]}"#).unwrap();
        let mut writer = apache_avro::Writer::new(&schema, &mut file);
        for sample in &test_data {
            writer
                .append(apache_avro::to_value(sample).unwrap())
                .unwrap();
        }
        writer.flush().unwrap();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        let (loaded_data, _) = load_regression_data(&file_path, 0.0).unwrap();
        let expected_data: Vec<Sample> = test_data
            .iter()
            .map(|sample| Sample::new(sample.features.clone(), sample.label))
            .collect();
        assert_eq!(expected_data, loaded_data);
    }
}
