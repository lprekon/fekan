#![feature(test)]
extern crate test;
use rand::{thread_rng, Rng};
use test::Bencher;

use fekan::kan_layer::{KanLayer, KanLayerOptions};

const INPUT_DIMENSION: usize = 128;
const OUTPUT_DIMENSION: usize = 12;
const DEGREE: usize = 5;
const COEF_SIZE: usize = 10;

fn build_test_layer() -> KanLayer {
    KanLayer::new(&KanLayerOptions {
        input_dimension: INPUT_DIMENSION,
        output_dimension: OUTPUT_DIMENSION,
        degree: DEGREE,
        coef_size: COEF_SIZE,
    })
}

#[bench]
fn bench_forward(b: &mut Bencher) {
    let mut layer = build_test_layer();
    let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show.
        for _ in 0..2 {
            let _ = layer.forward(&input);
        }
    });
}

#[bench]
fn bench_backward(b: &mut Bencher) {
    let mut layer = build_test_layer();
    let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    let _ = layer.forward(&input);
    let error = (0..OUTPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    b.iter(|| {
        // run multiple times per iteration so cache improvements will show
        for _ in 0..2 {
            let _ = layer.backward(&error);
        }
    });
}

#[bench]
fn bench_update(b: &mut Bencher) {
    let mut layer = build_test_layer();
    let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    let _ = layer.forward(&input);
    let error = (0..OUTPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    let _ = layer.backward(&error);
    b.iter(|| layer.update(0.1));
}

#[bench]
fn bench_update_knots_from_samples(b: &mut Bencher) {
    let mut layer = build_test_layer();
    for _ in 0..100 {
        let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
        let _ = layer.forward(&input);
    }

    b.iter(|| layer.update_knots_from_samples(0.1));
}

#[bench]
fn bench_forward_then_backward(b: &mut Bencher) {
    let mut layer = build_test_layer();
    let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    b.iter(|| {
        // no need for a loop - cached values from the forward pass should show up in the backward pass
        let output = layer.forward(&input).unwrap();
        let _ = layer.backward(&output);
    });
}
