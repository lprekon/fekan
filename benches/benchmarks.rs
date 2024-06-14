#![feature(test)]
extern crate test;
use rand::{thread_rng, Rng};
use test::Bencher;

use fekan::kan::kan_layer::KanLayer;

const INPUT_DIMENSION: usize = 128;
const OUTPUT_DIMENSION: usize = 12;
const DEGREE: usize = 5;
const COEF_SIZE: usize = 10;

#[bench]
fn bench_forward(b: &mut Bencher) {
    let mut layer = KanLayer::new(INPUT_DIMENSION, OUTPUT_DIMENSION, DEGREE, COEF_SIZE);
    let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    b.iter(|| layer.forward(&input));
}

#[bench]
fn bench_backward(b: &mut Bencher) {
    let mut layer = KanLayer::new(INPUT_DIMENSION, OUTPUT_DIMENSION, DEGREE, COEF_SIZE);
    let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    let _ = layer.forward(&input);
    let error = (0..OUTPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    b.iter(|| layer.backward(&error));
}

#[bench]
fn bench_update(b: &mut Bencher) {
    let mut layer = KanLayer::new(INPUT_DIMENSION, OUTPUT_DIMENSION, DEGREE, COEF_SIZE);
    let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    let _ = layer.forward(&input);
    let error = (0..OUTPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
    let _ = layer.backward(&error);
    b.iter(|| layer.update(0.1));
}

#[bench]
fn bench_update_knots_from_samples(b: &mut Bencher) {
    let mut layer = KanLayer::new(INPUT_DIMENSION, OUTPUT_DIMENSION, DEGREE, COEF_SIZE);
    for _ in 0..100 {
        let input = (0..INPUT_DIMENSION).map(|_| thread_rng().gen()).collect();
        let _ = layer.forward(&input);
    }

    b.iter(|| layer.update_knots_from_samples(0.1));
}
