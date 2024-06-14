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
