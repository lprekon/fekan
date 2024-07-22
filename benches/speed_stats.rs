#![feature(test)]
extern crate test;

use rand::{thread_rng, Rng};
use test::Bencher;

use fekan::kan_layer::{KanLayer, KanLayerOptions};

const DEGREE: usize = 5;

const INPUT_DIMENSION_BIG: usize = 128;
const INPUT_DIMENSION_SMALL: usize = 2;
const OUTPUT_DIMENSION_BIG: usize = 12;
const OUTPUT_DIMENSION_SMALL: usize = 2;

const COEF_SIZE_BIG: usize = 1000;
const COEF_SIZE_SMALL: usize = 10;

fn big_layer_small_spline() -> KanLayer {
    KanLayer::new(&KanLayerOptions {
        input_dimension: INPUT_DIMENSION_BIG,
        output_dimension: OUTPUT_DIMENSION_BIG,
        degree: DEGREE,
        coef_size: COEF_SIZE_SMALL,
    })
}

fn small_layer_big_spline() -> KanLayer {
    KanLayer::new(&KanLayerOptions {
        input_dimension: INPUT_DIMENSION_SMALL,
        output_dimension: OUTPUT_DIMENSION_SMALL,
        degree: DEGREE,
        coef_size: OUTPUT_DIMENSION_BIG,
    })
}

fn big_layer_big_spline() -> KanLayer {
    KanLayer::new(&KanLayerOptions {
        input_dimension: INPUT_DIMENSION_BIG,
        output_dimension: OUTPUT_DIMENSION_BIG,
        degree: DEGREE,
        coef_size: COEF_SIZE_BIG,
    })
}

#[bench]
fn bench_forward_big_layer_small_spline(b: &mut Bencher) {
    let mut layer = big_layer_small_spline();
    let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show.
        for _ in 0..2 {
            let _ = layer.forward(&input);
        }
    });
}

#[bench]
fn bench_forward_big_layer_big_spline(b: &mut Bencher) {
    let mut layer = big_layer_big_spline();
    let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show.
        for _ in 0..2 {
            let _ = layer.forward(&input);
        }
    });
}

#[bench]
fn bench_forward_small_layer_big_spline(b: &mut Bencher) {
    let mut layer = small_layer_big_spline();
    let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show.
        for _ in 0..2 {
            let _ = layer.forward(&input);
        }
    });
}

fn run_top_concurrent(b: &mut Bencher, mut layer: KanLayer, input_layer_size: usize) {
    let input: Vec<f64> = (0..input_layer_size).map(|_| thread_rng().gen()).collect();
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show.
        for _ in 0..2 {
            let _ = layer.forward_concurrent(&input, &thread_pool);
        }
    });
}

#[bench]
fn bench_forward_top_concurrent_big_layer_small_spline(b: &mut Bencher) {
    let layer = big_layer_small_spline();
    run_top_concurrent(b, layer, INPUT_DIMENSION_BIG);
}

#[bench]
fn bench_forward_top_concurrent_small_layer_big_spline(b: &mut Bencher) {
    let layer = small_layer_big_spline();
    run_top_concurrent(b, layer, INPUT_DIMENSION_SMALL)
}

#[bench]
fn bench_forward_top_concurrent_big_layer_big_spline(b: &mut Bencher) {
    let layer = big_layer_big_spline();
    run_top_concurrent(b, layer, INPUT_DIMENSION_BIG);
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
        for _ in 0..2 {
            let _ = layer.backward(&error);
        }
    });
}

#[bench]
fn bench_backward_concurrent(b: &mut Bencher) {
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
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
        for _ in 0..2 {
            let _ = layer.backward_concurrent(&error, &thread_pool);
        }
    });
}

#[bench]
fn bench_update(b: &mut Bencher) {
    let mut layer = big_layer_small_spline();
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
    let mut layer = big_layer_small_spline();
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
    let mut layer = big_layer_small_spline();
    for _ in 0..100 {
        let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
            .map(|_| thread_rng().gen())
            .collect();
        let _ = layer.forward(&input);
    }
    b.iter(|| layer.set_knot_length(100));
}

#[bench]
fn bench_forward_then_backward(b: &mut Bencher) {
    let mut layer = big_layer_small_spline();
    let input: Vec<f64> = (0..INPUT_DIMENSION_BIG)
        .map(|_| thread_rng().gen())
        .collect();
    b.iter(|| {
        // no need for a loop - cached values from the forward pass should show up in the backward pass
        let output = layer.forward(&input).unwrap();
        let _ = layer.backward(&output);
    });
}
