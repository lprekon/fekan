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

#[bench]
fn bench_forward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let inputs = vec![
        (0..INPUT_DIMENSION_BIG)
            .map(|_| thread_rng().gen())
            .collect::<Vec<f64>>();
        4 // create multiple inputs so we can see parallelism improvements later
    ];
    b.iter(|| {
        let _ = layer.forward(inputs.clone());
    });
}

#[bench]
fn bench_backward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let inputs = vec![
        (0..INPUT_DIMENSION_BIG)
            .map(|_| thread_rng().gen())
            .collect::<Vec<f64>>();
        4 // create multiple inputs so we can see parallelism improvements later
    ];
    let _ = layer.forward(inputs);
    let error = vec![
        (0..OUTPUT_DIMENSION_BIG)
            .map(|_| thread_rng().gen())
            .collect::<Vec<f64>>();
        4 // create multiple inputs so we can see parallelism improvements later
    ];
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show

        let _ = layer.backward(&error, 0.75, 0.25);
    });
}

#[bench]
fn bench_update(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let input = vec![(0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect()];
    let _ = layer.forward(input);
    let error = vec![(0..OUTPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect()];
    let _ = layer.backward(&error, 0.75, 0.25);
    b.iter(|| layer.update(0.1));
}

#[bench]
fn bench_update_knots_from_samples(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let mut batch_inputs = vec![vec![0.0; INPUT_DIMENSION_BIG]; 100];
    for i in 0..100 {
        for j in 0..INPUT_DIMENSION_BIG {
            batch_inputs[i][j] = thread_rng().gen();
        }
    }
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
