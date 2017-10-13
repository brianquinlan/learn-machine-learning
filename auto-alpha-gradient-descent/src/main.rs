extern crate rand;

use rand::{thread_rng, Rng};


fn dot_product(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn J(theta: &[f64; 4], x: &[[f64; 4]], y: &[f64]) -> f64 {
    let predictions = x.iter().map(|xi| dot_product(theta, xi));
    let error: f64 = predictions
        .zip(y.iter())
        .map(|(ti, yi)| (ti - yi).powi(2))
        .sum();
    1.0 / ((2 * x.len()) as f64) * error
}

fn main() {
    let mut rng = thread_rng();

    // Initialize the feature matrix.
    let mut x = [[0.0; 4]; 1000];
    for i in 0..x.len() {
        let xi = [
            1.0,
            rng.gen_range(-1000.0, 1000.0),
            rng.gen_range(-1000.0, 1000.0),
            rng.gen_range(-1000.0, 1000.0),
        ];
        x[i] = xi;
    }

    // Initialize the target vector according the the formula:
    //  3x^0 + 2x^1 + 8^x2 + 4x^3
    let mut y = [0.0; 1000];
    for i in 0..y.len() {
        y[i] = 3.0 + 2.0 * x[i][1] + 8.0 * x[i][2] + 4.0 * x[i][3];
    }

    // Run the gradient descent.
    let mut theta = [0.0; 4];
    let mut update = [0.0; 4];
    let mut alpha = 1.0;
    let mut error = J(&theta, &x, &y);
    for count in 0..100000000 {
        for j in 0..4 {
            let mut sum = 0.0;
            for i in 0..x.len() {
                let prediction: f64 = dot_product(&theta, &x[i]);
                sum += (prediction - y[i]) * x[i][j];
            }
            update[j] = theta[j] - alpha * (1.0 / (x.len() as f64)) * sum;
        }
        if theta == update {
            println!("Converged at iteration {:?} with alpha {:?}", count, alpha);
            break;
        }
        let new_error = J(&update, &x, &y);
        if new_error > error {
            alpha /= 3.0;
            println!("Reducing alpha to: {:?} at iteration {:?} [theta: {:?}]",
                alpha, count, theta);
        } else {
            theta = update;
            error = new_error;
        }
    }
    println!("{:?}", theta);
}
