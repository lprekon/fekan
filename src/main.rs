// use std::fs::File;

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    path::PathBuf,
};

// // use fekan::kan_layer::spline::Spline;
// // use fekan::kan_layer::KanLayer;
use clap::{CommandFactory, Parser, ValueEnum};

use fekan::{
    kan::{Kan, KanOptions, ModelType},
    train_model,
    training_observer::TrainingObserver,
    Sample, TrainingOptions,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

#[derive(Parser, Debug)]
struct Cli {
    #[arg(value_enum, long = "mode")]
    command: Commands,
    /// path to the parquet file containing the data
    #[arg(short = 'd', long = "data")]
    data_file: PathBuf,
    /// path to the model weights file.
    /// Only used for LoadTrain and LoadInfer commands
    #[arg(short = 'm', long = "model-in")]
    model_input_file: Option<PathBuf>,
    /// path to the output file for the model weights.
    /// Only used for Build and LoadTrain commands
    #[arg(short = 'o', long = "model-out")]
    model_output_file: Option<String>,

    /// how many samples to evaluate between knot updates.
    /// Only used for Build and LoadTrain commands
    #[arg(long, default_value = "100")]
    knot_update_interval: usize,

    /// when knot_adaptivity = 0, b-spline knots are uniformly distributed. When knot_adaptivity = 1, the knots are set using percentiles of the data. Values between 0 and 1 interpolate between these two extremes.
    #[arg(long, default_value = "0.1")]
    knot_adaptivity: f32,

    /// learning rate used to update the weights.
    /// Only used for Build and LoadTrain commands
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f32,

    /// number of epochs to train the model.
    /// Only used for Build and LoadTrain commands
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// fraction of the data to use for validation. Only used for Build and LoadTrain commands
    #[arg(long, default_value = "0.1")]
    validation_split: f32,

    /// if true, the model will be tested against the validation split after each epoch, and the validation loss included in the heartbeat.
    /// Only used for Build and LoadTrain commands
    #[arg(long)]
    validate_each_epoch: bool,

    /// degree of B-Spline to use.
    /// Only used for the Build command
    #[arg(short = 'k', long, default_value = "3")]
    degree: usize,

    /// number of control points to use in each B-Spline.
    /// Only used for the Build command
    #[arg(long, default_value = "5")]
    num_coef: usize,

    /// number of nodes in each interal layer of the model. This argument does not include the output layer.
    /// If not provided, the model will be a single layer with the same number of nodes as the number of classes.
    /// Only used for the Build command
    /// Ex `--hidden-layers 10,20,30` will create a model with 3 hidden layers, with 10, 20, and 30 nodes respectively
    #[arg(long = "hidden-layers", value_delimiter = ',')]
    hidden_layer_sizes: Option<Vec<usize>>,

    /// list of classes to predict, used to map output nodes to class labels, and to size the output layer in Build mode.
    /// In Build or LoadTrain mode, any data points with labels not in this list will be ignored
    /// Ex `--classes cat,dog,mouse`
    #[arg(long, value_delimiter = ',')]
    classes: Vec<String>,
}

// impl Into<TrainingOptions> for Cli {
//     fn into(self) -> TrainingOptions {
//         TrainingOptions {
//             num_epochs: self.epochs,
//             knot_update_interval: self.knot_update_interval,
//             learning_rate: self.learning_rate,
//             validate_each_epoch: self.validate_each_epoch,
//         }
//     }
// }

impl From<&Cli> for TrainingOptions {
    fn from(cli: &Cli) -> Self {
        TrainingOptions {
            num_epochs: cli.epochs,
            knot_update_interval: cli.knot_update_interval,
            learning_rate: cli.learning_rate,
            knot_adaptivity: cli.knot_adaptivity,
        }
    }
}

#[derive(ValueEnum, Clone, Debug)]
enum Commands {
    /// Build a new model and train it on the provided data
    Build,
    /// Load an existing model from a file and train it on the provided data
    LoadTrain,
    /// Load an existing model from a file and predict on the provided data
    LoadInfer,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    println!("Using arguments {cli:?}");
    match cli.command {
        Commands::Build => {
            if cli.model_output_file.is_none() {
                Cli::command()
                    .error(
                        clap::error::ErrorKind::MissingRequiredArgument,
                        "model output file is required for Build command",
                    )
                    .exit();
            }

            let (training_data, validation_data) =
                load_classification_data(&cli.data_file, cli.validation_split, &cli.classes)?;

            // if the user wants to validate the model after each epoch, pass the validation data to the training function as a signal to do so
            let to_pass_validation_data = if cli.validate_each_epoch {
                Some(&validation_data)
            } else {
                None
            };

            // build the model
            let input_dimension = training_data[0].features.len();
            let output_dimension = cli.classes.len();
            let mut layer_sizes: Vec<usize> = if cli.hidden_layer_sizes.is_some() {
                cli.hidden_layer_sizes.clone().unwrap()
            } else {
                vec![]
            };
            layer_sizes.push(output_dimension);

            let untrained_model = Kan::new(&KanOptions {
                input_size: input_dimension,
                layer_sizes,
                degree: cli.degree,
                coef_size: cli.num_coef,
                model_type: ModelType::Classification,
            });

            let observer = TrainingProgress::new((cli.epochs * training_data.len()) as u64);
            let trained_model = train_model(
                untrained_model,
                training_data,
                to_pass_validation_data,
                &observer,
                TrainingOptions::from(&cli),
            )?;
            observer.into_inner().finish();
            let mut out_file = File::create(cli.model_output_file.clone().unwrap())?;
            serde_pickle::to_writer(&mut out_file, &trained_model, Default::default()).unwrap();
            Ok(())
        }
        Commands::LoadTrain => {
            if cli.model_input_file.is_none() {
                Cli::command()
                    .error(
                        clap::error::ErrorKind::MissingRequiredArgument,
                        "model input file is required for LoadTrain command",
                    )
                    .exit();
            }
            if cli.model_output_file.is_none() {
                Cli::command()
                    .error(
                        clap::error::ErrorKind::MissingRequiredArgument,
                        "model output file is required for LoadTrain command",
                    )
                    .exit();
            }

            todo!("implement LoadTrain command");
        }
        Commands::LoadInfer => {
            if cli.model_input_file.is_none() {
                Cli::command()
                    .error(
                        clap::error::ErrorKind::MissingRequiredArgument,
                        "model input file is required for LoadInfer command",
                    )
                    .exit();
            }
            todo!("implement LoadInfer command")
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
