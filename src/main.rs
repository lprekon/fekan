// use fekan::kan_layer::spline::Spline;
// use fekan::kan_layer::KanLayer;
use fekan::Kan;

fn main() {
    let input_dimension = 64;
    let k = 3;
    let coef_size = 5;
    let layer_sizes = vec![128, 128, 128];
    let my_kan = Kan::new(input_dimension, layer_sizes.clone(), k, coef_size);
    println!(
        "A KAN with input dimension {}, layer sizes {:?}, k {}, and coef size {} has {} parameters",
        input_dimension,
        layer_sizes,
        k,
        coef_size,
        my_kan.get_parameter_count()
    );

    // let mut my_layer = KanLayer::new(input_dimension, output_dimension, k, coef_size);
    // print!("{:#?}", my_layer);
    // let preacts = vec![0.5, 0.4, 0.5];
    // let acts = my_layer.forward(preacts).unwrap();
    // println!("{:?}", acts);

    //     let knots = vec![0.0, 0.2857, 0.5714, 0.8571, 1.1429, 1.4286, 1.7143, 2.0];
    //     let control_points = vec![0.75, 1.0, 1.6, -1.0];
    //     let mut spline = Spline::new(3, control_points, knots).unwrap();
    //     let t = 0.975;
    //     //0.2535 + 0.5316 + 0.67664 - 0.0117
    //     let result = spline.forward(t);
}
