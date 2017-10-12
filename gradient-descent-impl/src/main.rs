extern crate rand;

use rand::{thread_rng, Rng};
use rand::distributions::{IndependentSample, Normal};


fn main() {
    let mut rng = thread_rng();

    // Initialize the feature matrix.
    let mut x = [[0.0; 4]; 1000];
    for i in 0..x.len() {
        let xi = [
            1.0,
            // Choosing a larger random range will increase the slope of the
            // resulting function so choosing a smaller alpha will be
            // necessary to prevent divergence.
            rng.gen_range(-10.0, 10.0),
            rng.gen_range(-10.0, 10.0),
            rng.gen_range(-10.0, 10.0),
        ];
        x[i] = xi;
    }

    // Initialize the target vector according the the formula:
    //  3x^0 + 2x^1 + 8^x2 + 4x^3
    // Use a distribution around the constants so that the error will
    // not be error.
    let mut y = [0.0; 1000];
    let a_distribution = Normal::new(3.0, 0.1);
    let b_distribution = Normal::new(2.0, 0.1);
    let c_distribution = Normal::new(8.0, 0.1);
    let d_distribution = Normal::new(4.0, 0.1);
    for i in 0..y.len() {
        let a = a_distribution.ind_sample(&mut rng);
        let b = b_distribution.ind_sample(&mut rng);
        let c = c_distribution.ind_sample(&mut rng);
        let d = d_distribution.ind_sample(&mut rng);

        let yi = a + b * x[i][1] + c * x[i][2] + d * x[i][3];
        y[i] = yi;
    }

    // Run the gradient descent.
    let mut theta = [0.0; 4];
    let mut update = [0.0; 4];
    let alpha = 0.00001;
    for count in 0..10000000 {
        for j in 0..4 {
            let mut sum = 0.0;
            for i in 0..x.len() {
                // The predicted value of y^i is:
                //  dot-product(transpose(theta), x^i).
                let prediction: f64 = theta.iter().zip(x[i].iter()).map(
                    |(x, y)| x * y).sum();
                sum += (prediction - y[i]) * x[i][j];
            }
            update[j] = theta[j] - alpha * (1.0 / (x.len() as f64)) * sum;
        }
        if theta == update {
            println!("Converged at iteration {:?}", count);
            break;
        }
        theta = update;
    }
    println!("{:?}", theta);
}
