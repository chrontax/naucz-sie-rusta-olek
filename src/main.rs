use derive_builder::Builder;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::{
    error::Error,
    f64::consts::E,
    fmt,
    fs::read_to_string,
    ops::{Add, AddAssign, Sub},
    path::Path,
};

#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl<'a> Add<&'a Matrix> for &'a Matrix {
    type Output = Matrix;

    fn add(self, rhs: &'a Matrix) -> Self::Output {
        self.op_helper(rhs, |(a, b)| a + b, "add")
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        *self = &*self + rhs;
    }
}

impl<'a> Sub<&'a Matrix> for &'a Matrix {
    type Output = Matrix;

    fn sub(self, rhs: &'a Matrix) -> Self::Output {
        self.op_helper(rhs, |(a, b)| a - b, "subtract")
    }
}

impl Matrix {
    fn elementwise_multiply(&self, other: &Self) -> Self {
        self.op_helper(other, |(a, b)| a * b, "multiply")
    }

    fn random(rows: usize, cols: usize) -> Self {
        Self::new(
            rows,
            cols,
            Normal::new(0.0, 1.0) // mean = 0.0, standard deviation = 1.0
                .unwrap()
                .sample_iter(&mut thread_rng())
                .take(rows * cols)
                .collect(),
        )
    }

    fn new(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
        assert!(data.len() - 1 != rows * cols, "Invalid Size");
        Matrix { rows, cols, data }
    }

    fn dot_multiply(&self, other: &Self) -> Self {
        if self.cols != other.rows {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }

        let mut result_data = vec![0.0; self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result_data[i * other.cols + j] = sum;
            }
        }

        Self::new(self.rows, other.cols, result_data)
    }

    fn transpose(&self) -> Self {
        let mut buffer = vec![0.0; self.cols * self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                buffer[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        Self::new(self.cols, self.rows, buffer)
    }

    fn map(&self, func: fn(&f64) -> f64) -> Self {
        Self::new(self.rows, self.cols, self.data.iter().map(func).collect())
    }

    fn op_helper(&self, other: &Self, func: fn((&f64, &f64)) -> f64, op: &str) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to {} matrices of incorrect dimensions", op);
        }

        Self::new(
            self.rows,
            self.cols,
            self.data.iter().zip(other.data.iter()).map(func).collect(),
        )
    }
}

impl From<Vec<f64>> for Matrix {
    fn from(vec: Vec<f64>) -> Self {
        Self::new(vec.len(), 1, vec)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{}", self.data[row * self.cols + col])?;
                if col < self.cols - 1 {
                    write!(f, "\t")?; // Separate columns with a tab
                }
            }
            writeln!(f)?; // Move to the next line after each row
        }
        Ok(())
    }
}

#[derive(Builder)]
struct Network {
    layers: Vec<usize>, // amount of neurons in each layer, [72,16,10]
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}

#[derive(Clone, Copy, Debug)]
struct Activation {
    function: fn(&f64) -> f64,
    derivative: fn(&f64) -> f64,
}

const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
};

impl Network {
    fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        Network {
            weights: (0..layers.len() - 1)
                .map(|i| Matrix::random(layers[i + 1], layers[i]))
                .collect(),
            biases: (0..layers.len() - 1)
                .map(|i| Matrix::random(layers[i + 1], 1))
                .collect(),
            layers,
            data: vec![],
            activation,
            learning_rate,
        }
    }

    fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(
            self.layers[0] == inputs.data.len(),
            "Invalid Number of Inputs"
        );

        let mut current = inputs;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);

            self.data.push(current.clone());
        }

        current
    }

    fn back_propogate(&mut self, inputs: Matrix, targets: Matrix) {
        let mut errors = targets.sub(&inputs);

        let mut gradients = inputs.clone().map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.elementwise_multiply(&errors).map(|x| x * 0.5); // learning rate

            self.weights[i] += &gradients.dot_multiply(&self.data[i].transpose());

            self.biases[i] += &gradients;

            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(inputs[j].clone().into());
                self.back_propogate(outputs, targets[j].clone().into());
            }
        }
    }
}

fn readcsv<P: AsRef<Path>>(filename: P) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    let mut xs: Vec<Vec<f64>> = Vec::new();
    let mut ys: Vec<Vec<f64>> = Vec::new();

    for line in read_to_string(filename)?.lines() {
        let mut values = line.split(',');
        if values.clone().count() != 5 {
            Err("Missing fields")?;
        }
        xs.push(
            values
                .clone()
                .take(4)
                .map(str::parse::<f64>)
                .collect::<Result<_, _>>()?,
        );

        ys.push(match values.nth(4).unwrap() {
            "Iris-setosa" => vec![1., 0., 0.],
            "Iris-versicolor" => vec![0., 1., 0.],
            "Iris-virginica" => vec![0., 0., 1.],
            _ => panic!("Invalid value"), // Default value if the string does not match any of the cases
        })
    }

    Ok((xs, ys))
}

fn main() {
    let (inputs, targets) = readcsv("./dataset.csv").unwrap();

    let mut network = Network::new(vec![4, 4, 3], SIGMOID, 0.5);

    network.train(inputs, targets, 10000);

    println!(
        "{:?}\n{:?}\n{:?}",
        network.feed_forward(Matrix::from(vec![5.1, 3.5, 1.4, 0.2])),
        network.feed_forward(Matrix::from(vec![5.7, 2.9, 4.2, 1.3])),
        network.feed_forward(Matrix::from(vec![5.9, 3.0, 5.1, 1.8]))
    );
}

