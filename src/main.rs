use std::fmt;
use std::ops::{Add, Mul};
use rand::Rng;
use std::f64::consts::E;
use std::env;
extern crate derive_builder;
use derive_builder::Builder;
use std::fs::File;
use std::path::Path;
use csv::Reader;
use std::{error::Error, io, process};
use rand_distr::{Normal, Distribution};
use rand::thread_rng;


#[derive(Debug,Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>
}


// access through  i* numofcols + j


impl Matrix {

    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
    
     if self.rows != other.rows || self.cols != other.cols {
			panic!("Attempted to multiply by matrix of incorrect dimensions");
		}

        let mut result_data = vec![0.0; self.cols * self.rows];
        for i in 0..self.data.len() { // double check this
            result_data[i] = self.data[i] * other.data[i]
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }
    
        pub fn random(rows: usize, cols: usize) -> Matrix {
            let mut buffer = Vec::<f64>::with_capacity(rows * cols);
            let normal = Normal::new(0.0, 1.0).unwrap(); // mean = 0.0, standard deviation = 1.0

            let mut rng = thread_rng();

            for _ in 0..rows * cols {
                let num = normal.sample(&mut rng);
                buffer.push(num);
            }

            Matrix { rows, cols, data: buffer }
        }



    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {

        assert!(data.len()-1 != rows * cols, "Invalid Size");
       Matrix { rows, cols, data }  
        
    }
   
    pub fn zeros(rows:usize, cols:usize) -> Matrix {

        Matrix { rows, cols, data: vec![0.0; cols * rows] }
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols {
			panic!("Attempted to add matrix of incorrect dimensions");
		}

      let mut buffer = Vec::<f64>::with_capacity(self.rows * self.cols);

      for i in 0..self.data.len() { 

              let result = self.data[i] + other.data[i];

              buffer.push(result);

      }

      Matrix { 
          rows:self.rows,
          cols: self.cols,
          data: buffer
      }

  }
  
    pub fn subtract(&self, other: &Matrix) -> Matrix {

        assert!(
          self.rows == other.rows && self.cols == other.cols,
          "Cannot subtract matrices with different dimensions"
      );

      let mut buffer = Vec::<f64>::with_capacity(self.rows * self.cols);

      for i in 0..self.data.len() { 

              let result = self.data[i] - other.data[i];

              buffer.push(result);

      }

      Matrix { 
          rows:self.rows,
          cols: self.cols,
          data: buffer
      }

  }
    
    
    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
       

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

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result_data,
        }
    
    }

    pub fn transpose(&self) -> Matrix {
        let mut buffer = vec![0.0; self.cols * self.rows];

        for i in 0..self.rows {
            for j in 0..self.cols {
                buffer[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: buffer,
        }
    }

    pub fn map(&mut self, func: fn(&f64) -> f64) -> Matrix
{
    let mut result = Matrix {
        rows: self.rows,
        cols: self.cols,
        data: Vec::with_capacity(self.data.len()),
    };

    result.data.extend(self.data.iter().map(|&val| func(&val)));

    result
}


}



impl From<Vec<f64>> for Matrix {
    fn from(vec: Vec<f64>) -> Self {
        let rows = vec.len();
        let cols = 1;
        Matrix {
            rows,
            cols,
            data: vec,
        }
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
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





#[derive( Builder)]
pub struct Network { 
    layers:Vec<usize>, // amount of neurons in each layer, [72,16,10]
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
}



#[derive(Clone,Copy,Debug)]
pub struct Activation {
    pub function: fn(&f64) -> f64,
    pub derivative: fn(&f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
};

impl Network {

    pub fn new(layers: Vec<usize>,activation:Activation,learning_rate:f64 ) -> Self { 

        let mut weights = vec![];

        let mut biases = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i+1], layers[i]));
            biases.push(Matrix::random(layers[i+1], 1));
        }


        Network { 
            layers, 
            weights, 
            biases, 
            data: vec![],
            activation,
            learning_rate
        }


    }

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {

        assert!(self.layers[0] == inputs.data.len(), "Invalid Number of Inputs");
     //   println!("{:?} {:?}",self.weights[0],inputs);
     //   println!("{:?}",self.weights[0].dot_multiply(&inputs).add(&self.biases[0]));
   
        let mut current = inputs;

            self.data = vec![current.clone()];


      for i in 0..self.layers.len() -1 {
            current = self.weights[i]
            .dot_multiply(&current)
            .add(&self.biases[i]).map(self.activation.function);
            
            self.data.push(current.clone());
      }


       current

    }

    pub fn back_propogate(&mut self, inputs:Matrix, targets:Matrix) {

        let mut errors = targets.subtract(&inputs);

        let mut gradients = inputs.clone().map(self.activation.derivative);


      

        for i in (0..self.layers.len() -1).rev(){
           
            gradients = gradients.elementwise_multiply(&errors).map(|x| x * 0.5); // learning rate
           
           
            
           
            self.weights[i] = self.weights[i].add(&gradients.dot_multiply(&self.data[i].transpose()));

            
            
        
            self.biases[i] = self.biases[i].add(&gradients);

            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);

        }      
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
		for i in 1..=epochs {
			if epochs < 100 || i % (epochs / 100) == 0 {
				println!("Epoch {} of {}", i, epochs);
			}
			for j in 0..inputs.len() {
				let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
				self.back_propogate(outputs,Matrix::from( targets[j].clone()));
			}
		}
	}




}


fn readcsv<P: AsRef<Path>>(filename: P) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    let file = File::open(filename)?;
    let mut rdr = Reader::from_reader(file);

    let mut studi_data: Vec<Vec<f64>> = Vec::new();
    let mut anser: Vec<Vec<f64>> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let to_add = vec![
            record.get(0).ok_or("Missing field")?.parse::<f64>()?,
            record.get(1).ok_or("Missing field")?.parse::<f64>()?,
            record.get(2).ok_or("Missing field")?.parse::<f64>()?,
            record.get(3).ok_or("Missing field")?.parse::<f64>()?,
        ];
        studi_data.push(to_add);

        let value_str = record.get(4).ok_or("Missing field")?;
        

        let value: f64 = match value_str {
            "Iris-setosa" => 0.25,
            "Iris-versicolor" => 0.5,
            "Iris-virginica" => 0.75,
            _ => 0.0, // Default value if the string does not match any of the cases
        };
        let val:Vec<f64> = vec!(value);
        anser.push(val);
    }

    

    Ok((studi_data, anser))
}


fn main() {
	env::set_var("RUST_BACKTRACE", "1");
    let Ok((inputs, targets)) = readcsv("./dataset.csv") else { todo!() };

    let mut network = Network::new(vec![4,4,1],SIGMOID,0.5);
    println!("{:?}",inputs);
   

    network.train(inputs, targets, 100000);

	println!("{:?}", network.feed_forward(Matrix::from(vec![5.1,3.5,1.4,0.2])));
    println!("{:?}", network.feed_forward(Matrix::from(vec![5.7,2.9,4.2,1.3])));
    println!("{:?}", network.feed_forward(Matrix::from(vec![5.9,3.0,5.1,1.8])));


    

}