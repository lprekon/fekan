#![feature(test)]
extern crate test;
use test::Bencher;

use fekan::kan::kan_layer::KanLayer;

#[bench]
fn bench_forward(b: &mut Bencher) {
    let mut layer = KanLayer::new(3, 2, 3, 4);
    let input = vec![1.0, 2.0, 3.0];
    b.iter(|| layer.forward(&input));
}

#[bench]
fn bench_backward(b: &mut Bencher) {
    let mut layer = KanLayer::new(3, 2, 3, 4);
    let input = vec![1.0, 2.0, 3.0];
    let _ = layer.forward(&input);
    let error = vec![1.0, 2.0];
    b.iter(|| layer.backward(&error));
}

#[bench]
fn bench_update(b: &mut Bencher) {
    let mut layer = KanLayer::new(3, 2, 3, 4);
    let input = vec![1.0, 2.0, 3.0];
    let _ = layer.forward(&input);
    let error = vec![1.0, 2.0];
    let _ = layer.backward(&error);
    b.iter(|| layer.update(0.1));
}

#[bench]
fn bench_update_knots_from_samples(b: &mut Bencher) {
    let mut layer = KanLayer::new(3, 2, 3, 4);
    let input = vec![1.0, 2.0, 3.0];
    for _ in 0..100 {
        let _ = layer.forward(&input);
        let error = vec![1.0, 2.0];
        let _ = layer.backward(&error);
    }

    b.iter(|| layer.update_knots_from_samples(0.1));
}
