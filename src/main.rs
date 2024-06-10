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
use fekan::Kan;
use serde::Deserialize;

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

#[derive(ValueEnum, Clone)]
enum Commands {
    /// Build a new model and train it on the provided data
    Build,
    /// Load an existing model from a file and train it on the provided data
    LoadTrain,
    /// Load an existing model from a file and predict on the provided data
    LoadInfer,
}

#[derive(Deserialize, Debug)]
struct RawSample {
    features: Vec<u8>,
    label: String,
}

#[derive(Clone)]
struct Sample {
    features: Vec<u8>,
    label: u8,
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
            let model = build_and_train(cli)?;
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

static LABELS: &'static [&str] = &[
    "avr",
    "alphaev56",
    "arm",
    "m68k",
    "mips",
    "mipsel",
    "powerpc",
    "s390",
    "sh4",
    "sparc",
    "x86_64",
    "xtensa",
];

fn load_data(args: &Cli) -> (Vec<Sample>, Vec<Sample>) {
    let file = File::open(args.data_file.clone()).unwrap();
    let raw_data: Vec<RawSample> = serde_pickle::from_reader(file, Default::default()).unwrap();
    let rows_loaded = raw_data.len();
    let mut label_map: HashMap<&str, u8> = HashMap::new();
    for (i, label) in LABELS.iter().enumerate() {
        label_map.insert(label, i as u8);
    }

    // parse the data, maping the label string to a u8
    let data = raw_data
        .iter()
        .map(|r| Sample {
            features: r.features.clone(),
            label: label_map[&r.label[..]],
        })
        .collect::<Vec<Sample>>();

    // separate the data into training and validation sets
    let mut validation_indecies: HashSet<usize> = HashSet::new();
    while validation_indecies.len() < (args.validation_split * data.len() as f32) as usize {
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
        "data split error",
    );
    (training_data, validation_data)
}

fn build_and_train(args: Cli) -> Result<Kan, Box<dyn Error>> {
    /* 1. Load the data
       1a. PARSE the data
    * 2. Separate the data into training and validation sets
    * 3. Build the model
    * 4. Run the training loop
    * 5. Save the model
    */

    // load, parse, and separate the data
    let (training_data, validation_data) = load_data(&args);

    // build the model
    let input_dimension = training_data[0].features.len();
    let k = 3;
    let coef_size = 5;
    let layer_sizes = vec![LABELS.len(), LABELS.len()];
    let mut model = Kan::new(input_dimension, layer_sizes, k, coef_size);
    println!("model parameter count: {}", model.get_parameter_count());

    // train the model
    for epoch in 0..args.epochs {
        let mut epoch_loss = 0.0;
        let mut samples_seen = 0;
        for sample in &training_data {
            samples_seen += 1;
            // run over each sample in the training data for each epoch
            let logits = model
                .forward(sample.features.iter().map(|&x| x as f32).collect())
                .unwrap();
            // calculate classification probability from logits
            let (loss, dlogits) = calculate_error(&logits, sample.label as usize);
            epoch_loss += loss;
            // pass the error back through the model
            let _ = model.backward(dlogits).unwrap();
            model.update(args.learning_rate); // TODO implement momentum
            model.zero_gradients();
            if samples_seen % args.knot_update_interval == 0 {
                let _ = model.update_knots_from_samples()?;
            }
        }
        epoch_loss /= training_data.len() as f32;

        let mut validation_loss = 0.0;
        if args.validate_each_epoch {
            for sample in &validation_data {
                let logits = model
                    .forward(sample.features.iter().map(|&x| x as f32).collect())
                    .unwrap();
                let (loss, _) = calculate_error(&logits, sample.label as usize);
                validation_loss += loss;
            }
            validation_loss /= validation_data.len() as f32;
        }
        // print stats
        print!(
            "[HEARTBEAT] {} epoch: {} epoch_loss: {}",
            chrono::Local::now(),
            epoch,
            epoch_loss
        );
        if args.validate_each_epoch {
            print!(" validation_loss: {validation_loss}");
        }
        println!();
    }

    Ok(model)
}

fn calculate_error(logits: &Vec<f32>, label: usize) -> (f32, Vec<f32>) {
    // calculate the classification probabilities
    let (logit_max, logit_max_index) = {
        let mut max = f32::NEG_INFINITY;
        let mut max_index = 0;
        for (i, &logit) in logits.iter().enumerate() {
            if logit > max {
                max = logit;
                max_index = i;
            }
        }
        (max, max_index)
    };
    let norm_logits = logits.iter().map(|&x| x - logit_max).collect::<Vec<f32>>(); // subtract the max logit to prevent overflow
    let counts = norm_logits.iter().map(|&x| x.exp()).collect::<Vec<f32>>();
    let count_sum = counts.iter().sum::<f32>();
    let probs = counts.iter().map(|&x| x / count_sum).collect::<Vec<f32>>();

    // calculate the loss
    let logprobs = probs.iter().map(|&x| x.ln()).collect::<Vec<f32>>();
    let loss = -logprobs[label];

    // calculate the error
    let dlogprobs = (0..probs.len())
        .map(|i| if i == label { -1.0 } else { 0.0 })
        .collect::<Vec<f32>>(); // dloss/dlogpobs. vector is 0 except for the correct class, where it's -1
    let dprobs = probs
        .iter()
        .zip(dlogprobs.iter())
        .map(|(&p, &dlp)| dlp / p)
        .collect::<Vec<f32>>(); // dloss/dprobs = dlogprobs/dprobs * dloss/dlogprobs. d/dx ln(x) = 1/x. dlogprobs/dprobs = 1/probs, `dlp` = dloss/dlogprobs
    let dcounts_sum: f32 = counts
        .iter()
        .zip(dprobs.iter())
        .map(|(&count, &dprob)| -count / (count_sum * count_sum) * dprob)
        .sum();
    let dcounts = dprobs
        .iter()
        .map(|&dprob| dcounts_sum + dprob / count_sum)
        .collect::<Vec<f32>>();
    let dnorm_logits = dcounts
        .iter()
        .zip(counts.iter())
        .map(|(&dcount, &e_norm_logit)| dcount * e_norm_logit)
        .collect::<Vec<f32>>(); // dloss/dnorm_logits = dloss/dcounts * dcounts/dnorm_logits, dcounts/dnorm_logits = d/dx exp(x) = exp(x), and exp(norm_logits) = counts, so we just use counts rather than recalculating
    let dlogit_max: f32 = -dnorm_logits.iter().sum::<f32>();
    let dlogits = dnorm_logits.iter().enumerate().map(|(i, &dnorm_logit)|{if i == logit_max_index {1.0} else {0.0}} * dlogit_max + dnorm_logit).collect::<Vec<f32>>();

    (loss, dlogits)
}
