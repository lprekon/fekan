#![feature(test)]
extern crate test;

use rand::{thread_rng, Rng};
use test::Bencher;

use fekan::kan_layer::{KanLayer, KanLayerOptions};

const DEGREE: usize = 5;

const INPUT_DIMENSION_BIG: usize = 128;
// const INPUT_DIMENSION_SMALL: usize = 2;
const OUTPUT_DIMENSION_BIG: usize = 12;
// const OUTPUT_DIMENSION_SMALL: usize = 2;

const COEF_SIZE_BIG: usize = 100;
// const COEF_SIZE_SMALL: usize = 10;

// fn big_layer_small_spline() -> KanLayer {
//     KanLayer::new(&KanLayerOptions {
//         input_dimension: INPUT_DIMENSION_BIG,
//         output_dimension: OUTPUT_DIMENSION_BIG,
//         degree: DEGREE,
//         coef_size: COEF_SIZE_SMALL,
//     })
// }

// fn small_layer_big_spline() -> KanLayer {
//     KanLayer::new(&KanLayerOptions {
//         input_dimension: INPUT_DIMENSION_SMALL,
//         output_dimension: OUTPUT_DIMENSION_SMALL,
//         degree: DEGREE,
//         coef_size: OUTPUT_DIMENSION_BIG,
//     })
// }

fn big_layer_big_spline() -> KanLayer {
    KanLayer::new(&KanLayerOptions {
        input_dimension: INPUT_DIMENSION_BIG,
        output_dimension: OUTPUT_DIMENSION_BIG,
        degree: DEGREE,
        coef_size: COEF_SIZE_BIG,
    })
}

const BATCH_SIZE: usize = 4;

fn generate_batch_inputs() -> Vec<Vec<f64>> {
    let mut batch_inputs = vec![vec![0.0; INPUT_DIMENSION_BIG]; BATCH_SIZE];
    for i in 0..BATCH_SIZE {
        // create multiple samples so we can see parallelism improvements later
        for j in 0..INPUT_DIMENSION_BIG {
            batch_inputs[i][j] = thread_rng().gen();
        }
    }
    batch_inputs
}

fn generate_batch_gradients() -> Vec<Vec<f64>> {
    let mut batch_gradients = vec![vec![0.0; OUTPUT_DIMENSION_BIG]; BATCH_SIZE];
    for i in 0..BATCH_SIZE {
        // create multiple samples so we can see parallelism improvements later
        for j in 0..OUTPUT_DIMENSION_BIG {
            batch_gradients[i][j] = thread_rng().gen();
        }
    }
    batch_gradients
}

#[bench]
fn bench_forward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    b.iter(|| {
        let _ = layer.forward(batch_inputs.clone());
    });
}

#[bench]
fn bench_one_threaded_forward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    b.iter(|| {
        let _ = layer.forward_multithreaded(batch_inputs.clone(), 1);
    });
}

#[bench]
fn bench_two_threaded_forward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    b.iter(|| {
        let _ = layer.forward_multithreaded(batch_inputs.clone(), 2);
    });
}

#[bench]
fn bench_four_threaded_forward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    b.iter(|| {
        let _ = layer.forward_multithreaded(batch_inputs.clone(), 4);
    });
}

#[bench]
fn bench_eight_threaded_forward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    b.iter(|| {
        let _ = layer.forward_multithreaded(batch_inputs.clone(), 8);
    });
}

#[bench]
fn bench_backward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    let _ = layer.forward(batch_inputs);
    let batch_gradients = generate_batch_gradients();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show

        let _ = layer.backward(&batch_gradients);
    });
}

#[bench]
fn bench_update(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    let _ = layer.forward(batch_inputs);
    let error = vec![(0..OUTPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect()];
    let _ = layer.backward(&error);
    b.iter(|| layer.update(0.1, 0.75, 0.25));
}

#[bench]
fn bench_update_knots_from_samples(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let batch_inputs = generate_batch_inputs();
    let _ = layer.forward(batch_inputs);

    b.iter(|| layer.update_knots_from_samples(0.1));
}

#[bench]
fn bench_set_knot_length(b: &mut Bencher) {
    let mut layer = KanLayer::new(&KanLayerOptions {
        input_dimension: 1,
        output_dimension: 1,
        degree: 3,
        coef_size: COEF_SIZE_BIG,
    });

    b.iter(|| layer.set_knot_length(COEF_SIZE_BIG * 2));
}

#[bench]
fn bench_suggest_symbolic(b: &mut Bencher) {
    let layer = KanLayer::new(&KanLayerOptions {
        input_dimension: 1,
        output_dimension: 1,
        degree: 3,
        coef_size: 10,
    });
    b.iter(|| {
        let _ = layer.bench_suggest_symbolic();
    });
}

#[bench]
fn bench_prune(b: &mut Bencher) {
    let mut layer = KanLayer::new(&KanLayerOptions {
        input_dimension: 1,
        output_dimension: 1,
        degree: 3,
        coef_size: 10,
    });
    let mut batch_inputs = vec![vec![0.0; INPUT_DIMENSION_BIG]; 100]; // generate 100 samples to we're feeding the same amount through we used to before the refactor
    for i in 0..100 {
        // create multiple samples so we can see parallelism improvements later
        for j in 0..INPUT_DIMENSION_BIG {
            batch_inputs[i][j] = thread_rng().gen();
        }
    }
    b.iter(|| {
        let _ = layer.prune(&batch_inputs, 0.0);
    });
}
