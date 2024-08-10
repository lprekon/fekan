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
    let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show.

        let _ = layer.forward(&input);
    });
}

#[bench]
fn bench_backward(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    let _ = layer.forward(&input);
    let error: Vec<f64> = (0..OUTPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show

        let _ = layer.backward(&error);
    });
}

#[bench]
fn bench_update(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    let _ = layer.forward(&input);
    let error: Vec<f64> = (0..OUTPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    let _ = layer.backward(&error);
    b.iter(|| layer.update(0.1));
}

#[bench]
fn bench_update_knots_from_samples(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    for _ in 0..100 {
        let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
            .map(|_| thread_rng().gen())
            .collect();
        let _ = layer.forward(&input);
    }

    b.iter(|| layer.update_knots_from_samples(0.1));
}

#[bench]
fn bench_set_knot_length(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();

    b.iter(|| layer.set_knot_length(COEF_SIZE_BIG * 3));
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
