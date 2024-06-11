// use std::fs::File;

use std::{error::Error, fs::File, path::PathBuf};

// // use fekan::kan_layer::spline::Spline;
// // use fekan::kan_layer::KanLayer;
use clap::{CommandFactory, Parser, ValueEnum};

use fekan::{build_and_train, TrainingOptions};

#[derive(Parser)]
struct Cli {
    #[arg(value_enum, long = "mode")]
    command: Commands,
    /// path to the parquet file containing the data
    #[arg(short = 'd', long = "data")]
    data_file: PathBuf,
    /// path to the model weights file. Only used for LoadTrain and LoadInfer commands
    #[arg(short = 'm', long = "model-in")]
    model_input_file: Option<PathBuf>,
    /// path to the output file for the model weights. Only used for Build and LoadTrain commands
    #[arg(short = 'o', long = "model-out")]
    model_output_file: Option<String>,

    /// how many rows to evaluate between knot updates
    #[arg(short, long, default_value = "100")]
    knot_update_interval: usize,

    // TODO implement the knot adaptivity parameter
    /// at `0`, the knots are evenly spaced. At `1`, the knots are denser where the samples are denser. values between `0` and `1` interpolate between the two
    // #[arg(long, default_value = "0.2")]
    // knot_adaptivity: f32,

    /// learning rate used to update the weights
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f32,

    /// number of epochs to train the model
    #[arg(short, long, default_value = "100")]
    epochs: usize,

    /// fraction of the data to use for validation
    #[arg(long, default_value = "0.1")]
    validation_split: f32,

    validate_each_epoch: bool,
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
            validate_each_epoch: cli.validate_each_epoch,
        }
    }
}

#[derive(ValueEnum, Clone)]
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

            let mut out_file = File::create(cli.model_output_file.clone().unwrap())?;
            let (training_data, validation_data) =
                fekan::load_data(&cli.data_file, cli.validation_split)?;
            let model =
                build_and_train(training_data, validation_data, TrainingOptions::from(&cli))?;
            serde_pickle::to_writer(&mut out_file, &model, Default::default()).unwrap();
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
