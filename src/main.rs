// use std::fs::File;

use std::{fs::File, path::PathBuf};

// // use fekan::kan_layer::spline::Spline;
// // use fekan::kan_layer::KanLayer;
// use fekan::Kan;
use clap::{CommandFactory, Parser, ValueEnum};

#[derive(Parser)]
struct Cli {
    #[arg(value_enum, long = "mode")]
    command: Commands,
    /// path to the parquet file containing the data
    #[arg(short = 'd', long = "data")]
    data_file: String,
    /// path to the model weights file. Only used for LoadTrain and LoadInfer commands
    #[arg(short = 'm', long = "model-in")]
    model_input_file: Option<PathBuf>,
    /// path to the output file for the model weights. Only used for Build and LoadTrain commands
    #[arg(short = 'o', long = "model-out")]
    model_output_file: Option<String>,
    /// number of rows to pass through the model per batch. Weights knots are updated after each batch
    #[arg(long, default_value = "1024")]
    batch_size: usize,

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

    /// number of batches to process before printing the loss and other metrics
    #[arg(long, default_value = "10")]
    batches_per_heartbeat: usize,

    /// fraction of the data to use for validation
    #[arg(long, default_value = "0.1")]
    validation_split: f32,
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

struct DataSample {
    features: Vec<u8>,
    target: u8,
}

fn main() {
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
            build_and_train(cli)
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

fn build_and_train(args: Cli) {
    /* 1. Load the data
       1a. PARSE the data
    * 2. Separate the data into training and validation sets
    * 3. Build the model
    * 4. Run the training loop
    * 5. Save the model
    */

    let file = File::open(args.data_file).unwrap();
}
