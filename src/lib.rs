pub mod kan;

use kan::Kan;
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    path::PathBuf,
};

#[derive(Deserialize, Debug)]
struct RawSample {
    features: Vec<u8>,
    label: String,
}

#[derive(Clone)]
pub struct Sample {
    features: Vec<u8>,
    label: u8,
}

pub struct TrainingOptions {
    pub num_epochs: usize,
    pub knot_update_interval: usize,
    pub learning_rate: f32,
    pub validate_each_epoch: bool,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        TrainingOptions {
            num_epochs: 100,
            knot_update_interval: 100,
            learning_rate: 0.001,
            validate_each_epoch: false,
        }
    }
}

pub fn build_and_train(
    training_data: Vec<Sample>,
    validation_data: Vec<Sample>,
    options: TrainingOptions,
) -> Result<Kan, Box<dyn Error>> {
    /* 1. Load the data
       1a. PARSE the data
    * 2. Separate the data into training and validation sets
    * 3. Build the model
    * 4. Run the training loop
    * 5. Save the model
    */

    // load, parse, and separate the data
    // let (training_data, validation_data) = load_data(data_file_path, validation_split);

    // build the model
    let input_dimension = training_data[0].features.len();
    let k = 3;
    let coef_size = 5;
    let layer_sizes = vec![LABELS.len(), LABELS.len()];
    let mut model = Kan::new(input_dimension, layer_sizes, k, coef_size);
    println!("model parameter count: {}", model.get_parameter_count());

    // train the model
    for epoch in 0..options.num_epochs {
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
            model.update(options.learning_rate); // TODO implement momentum
            model.zero_gradients();
            if samples_seen % options.knot_update_interval == 0 {
                let _ = model.update_knots_from_samples()?;
            }
        }
        epoch_loss /= training_data.len() as f32;

        let mut validation_loss = 0.0;
        if options.validate_each_epoch {
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
        if options.validate_each_epoch {
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

pub fn load_data(
    data_file_path: &PathBuf,
    validation_split: f32,
) -> Result<(Vec<Sample>, Vec<Sample>), Box<dyn Error>> {
    let file = File::open(data_file_path)?;
    let raw_data: Vec<RawSample> = serde_pickle::from_reader(file, Default::default())?;
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
        "data split error",
    );
    Ok((training_data, validation_data))
}
