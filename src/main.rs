// use std::fs::File;

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    path::PathBuf,
};

// // use fekan::kan_layer::spline::Spline;
// // use fekan::kan_layer::KanLayer;
use clap::{Args, Parser, Subcommand};

use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model,
    training_observer::TrainingObserver,
    Sample, TrainingOptions,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

#[derive(Parser, Debug, Clone)]
struct Cli {
    #[command(subcommand)]
    command: WhereCommands,
    /// path to the file containing the data. currently only supports pickle files
    #[arg(short = 'd', long = "data", required = true)]
    data_file: PathBuf,
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
    /// A comma-separated list of classes which the model will be trained to predict. The model will have ||classes|| output nodes
    #[arg(short, long, required = true, value_delimiter = ',')]
    classes: Vec<String>,

    #[command(flatten)]
    params: GenericBuildParams,
}

#[derive(Args, Clone, Debug)]
struct RegressorArgs {
    #[arg(long = "output-nodes", default_value = "1")]
    /// The number of output nodes in the model. If > 1, the model will be a multi-output regression model
    num_output_nodes: usize,
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

    #[arg(long, alias = "hl", global = true)]
    /// a comma-separated list of hidden layer sizes. If empty, the model will only have the output layer
    hidden_layer_sizes: Option<Vec<usize>>,
    // #[command(flatten)]
    #[command(flatten)]
    training_parameters: TrainArgs,
}

#[derive(Args, Clone, Debug)]
struct TrainArgs {
    #[arg(short = 'e', long, default_value = "100", global = true)]
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
    knot_adaptivity: f32,

    #[arg(long, alias = "lr", default_value = "0.01", global = true)]
    /// the learning rate used to update the model weights
    learning_rate: f32,

    #[arg(long, global = true)]
    /// if set, the model will be run against the validation data after each epoch, and the loss will be reported to the observer
    validate_each_epoch: bool,

    #[arg(long, default_value = "0.2", global = true)]
    /// the fraction of the training data to use as validation data
    validation_split: f32,

    /// path to the output file for the model weights.
    #[arg(short = 'o', long = "model-out", required = true)]
    model_output_file: PathBuf,
}

#[derive(Args, Clone, Debug)]
struct LoadArgs {
    /// path to the model weights file.
    model_input_file: PathBuf,
    #[command(subcommand)]
    command: WhyCommands,
}

#[derive(Subcommand, Debug, Clone)]
enum WhyCommands {
    /// Further train models weights loaded from a file
    Train(TrainArgs),
    /// Use a loaded model to make predictions on new data
    Infer {
        /// an ordered list of human-readable labels for the output nodes of the model
        class_map: Vec<String>,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    println!("Using arguments {cli:?}");
    match cli.command {
        WhereCommands::Build(build_args) => match build_args.model_type {
            CliModelType::Classifier(classifier_args) => {
                let (training_data, validation_data) = load_classification_data(
                    &cli.data_file,
                    classifier_args.params.training_parameters.validation_split,
                    &classifier_args.classes,
                )?;

                let observer_ticks = if classifier_args
                    .params
                    .training_parameters
                    .validate_each_epoch
                {
                    training_data.len() * classifier_args.params.training_parameters.num_epochs
                } else {
                    training_data.len() + validation_data.len()
                };
                let training_observer = TrainingProgress::new(observer_ticks as u64);

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
                    input_size: training_data[0].features.len() as usize,
                    layer_sizes: layers,
                    degree: classifier_args.params.degree,
                    coef_size: classifier_args.params.num_coefficients,
                    model_type: ModelType::Classification,
                });

                let training_options = TrainingOptions {
                    num_epochs: classifier_args.params.training_parameters.num_epochs,
                    knot_update_interval: classifier_args
                        .params
                        .training_parameters
                        .knot_update_interval,
                    knot_adaptivity: classifier_args.params.training_parameters.knot_adaptivity,
                    learning_rate: classifier_args.params.training_parameters.learning_rate,
                };

                // if the user wants the model validated each epoch, pass the validation data to the training function.
                // Otherwise, pass None
                let passed_validation_data = if classifier_args
                    .params
                    .training_parameters
                    .validate_each_epoch
                {
                    Some(&validation_data)
                } else {
                    None
                };
                // run the training loop on the model
                let trained_model = train_model(
                    untrained_model,
                    training_data,
                    passed_validation_data,
                    &training_observer,
                    training_options,
                )?;
                training_observer.into_inner().finish();
                // save the model to a file
                let mut out_file =
                    File::create(classifier_args.params.training_parameters.model_output_file)?;
                serde_pickle::to_writer(&mut out_file, &trained_model, Default::default()).unwrap();
                Ok(())
            }
            CliModelType::Regressor(_regressor_args) => {
                todo!("Reimplement the regressor model building")
            }
        },
        WhereCommands::Load(_load_args) => {
            todo!("Reimplement the load command")
        }
    }
}

struct TrainingProgress {
    pb: ProgressBar,
}

impl TrainingProgress {
    fn new(total: u64) -> Self {
        let pb = ProgressBar::new(total);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] [{bar:40.green/white}] {human_pos}/{human_len} {per_sec} ({eta})",
                )
                .unwrap(),
        );
        TrainingProgress { pb }
    }

    fn into_inner(self) -> ProgressBar {
        self.pb
    }
}

impl TrainingObserver for TrainingProgress {
    fn on_sample_end(&self) {
        self.pb.inc(1);
    }
    fn on_epoch_end(&self, epoch: usize, epoch_loss: f32, validation_loss: f32) {
        self.pb.println(format!(
            "{} Epoch {}: Training Loss: {}, Validation Loss: {}",
            chrono::Local::now(),
            epoch,
            epoch_loss,
            validation_loss
        ));
    }
}

#[derive(Deserialize, Debug)]
struct RawSample {
    features: Vec<u8>,
    label: String,
}

pub fn load_classification_data(
    data_file_path: &PathBuf,
    validation_split: f32,
    classes: &Vec<String>,
) -> Result<(Vec<Sample>, Vec<Sample>), Box<dyn Error>> {
    let file = File::open(data_file_path)?;
    let raw_data: Vec<RawSample> = serde_pickle::from_reader(file, Default::default())?;
    let rows_loaded = raw_data.len();

    // parse the data, maping the label string to a u8
    let class_map: HashMap<String, u32> = HashMap::from_iter(
        classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.clone(), i as u32)),
    );
    println!("Class map: {:?}", class_map);
    // let data: Vec<Sample> = raw_data.iter().filter(|raw_sample| class_map.contains_key(&raw_sample.label)).map(|raw_sample| {
    //     let label = class_map[&raw_sample.label];
    //     let features: Vec<f32> = raw_sample.features.iter().map(|&f| f as f32).collect();
    //     Sample { features, label }
    // }).collect();
    let mut data = Vec::with_capacity(raw_data.len());
    for raw_sample in raw_data {
        if !class_map.contains_key(&raw_sample.label) {
            continue; // ignore data points with labels not in the provided class list
        }
        let label = class_map[&raw_sample.label];
        let features: Vec<f32> = raw_sample.features.iter().map(|&f| f as f32).collect();
        data.push(Sample {
            features,
            label: label as f32,
        });
    }

    // separate the data into training and validation sets
    let mut validation_indecies: HashSet<usize> = HashSet::new();
    while validation_indecies.len() < (validation_split * data.len() as f32) as usize {
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
