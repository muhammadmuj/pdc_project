# Apache spark 
# Apache Spark MLlib: Iris Dataset Classification

## Overview
This project demonstrates the use of **Apache Spark MLlib** for classifying the Iris dataset using **Logistic Regression**. The implementation highlights key aspects such as **scalability**, **fault tolerance**, **memory usage tracking**, and **performance evaluation**.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Technologies Used](#technologies-used)
4. [Implementation Steps](#implementation-steps)
5. [Performance Metrics](#performance-metrics)
6. [Visualization](#visualization)
7. [Fault Tolerance Simulation](#fault-tolerance-simulation)
8. [How to Run](#how-to-run)
9. [Results](#results)
10. [License](#license)

---

## Introduction
Apache Spark MLlib is a scalable machine learning library built on Apache Spark. This project utilizes Spark MLlib to classify the Iris dataset using Logistic Regression, demonstrating Spark's ability to handle distributed data processing, scalability, and fault tolerance.

---

## Dataset
- **Dataset Name:** Iris Dataset
- **Source:** Scikit-learn library
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width

---

## Technologies Used
- **Apache Spark MLlib**
- **Python**
- **Scikit-learn**
- **Matplotlib**
- **psutil** (for memory tracking)

---

## Implementation Steps
1. **Initialize Spark Session:** Create a local Spark session.
2. **Load Dataset:** Use Scikit-learn to load and prepare the Iris dataset.
3. **Data Preprocessing:** Use `VectorAssembler` to prepare feature columns.
4. **Train-Test Split:** Split data into 80% training and 20% testing.
5. **Model Training:** Train a Logistic Regression model.
6. **Fault Tolerance:** Simulate and handle training faults.
7. **Model Evaluation:** Evaluate accuracy using `MulticlassClassificationEvaluator`.
8. **Performance Metrics:** Track training time, inference time, and memory usage.
9. **Visualization:** Plot key metrics.

---

## Performance Metrics
- **Training Time:** Time taken to train the model.
- **Inference Time:** Time taken for predictions.
- **Memory Usage:** Memory before and after training/inference.
- **Accuracy:** Model accuracy on the test dataset.

---

## Visualization
A bar plot is generated to display:
- Training Time
- Inference Time
- Memory Before Training
- Memory After Inference
- Fault Recovery Time

---

## Fault Tolerance Simulation
A fault is intentionally triggered during the training phase. The model recovers gracefully and resumes training without data loss.

---

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pyspark scikit-learn matplotlib psutil
   ```
3. Run the script:
   ```bash
   python script_name.py
   ```
4. Results and visualizations will be displayed in the terminal and plots.

---

## Results
- Model achieved an accuracy of approximately **90%**.
- Fault recovery was successful.
- Scalability and memory usage were analyzed and visualized.

---

## License
This project is licensed under the **MIT License**.

---

## Author
**Muhammad Mujtaba**

contact: **muj86218@gmail.com**

# Ray and Tensorflow
# Distributed MNIST Training with Ray and TensorFlow

## Overview
This project demonstrates distributed training and fault tolerance using **Ray** and **TensorFlow** on the **MNIST dataset**. The implementation showcases scalability, resource monitoring, and fault tolerance capabilities across multiple workers.

### Key Features:
- **Distributed Training:** Parallel training using multiple Ray workers.
- **Fault Tolerance:** Simulates worker failures and recovers gracefully.
- **Resource Monitoring:** Tracks CPU and memory usage.
- **Scalability Testing:** Measures performance with varying numbers of workers.

---

## Prerequisites
Ensure the following dependencies are installed:

- Python 3.8+
- TensorFlow 2.x
- Ray
- NumPy
- Psutil

Install dependencies using pip:
```bash
pip install tensorflow ray numpy psutil
```

---

## Project Structure
- **main.py**: Core script for distributed training, scalability, and fault tolerance.
- **README.md**: Documentation.

---

## Dataset
- **MNIST Dataset:** Handwritten digit dataset with 60,000 training images and 10,000 test images.
- Automatically downloaded using TensorFlow's `mnist.load_data()`.

---

## How to Run

1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>
```

2. Run the script:
```bash
python main.py
```

### Expected Outputs:
- Training and Inference Time
- Average Accuracy across workers
- Resource Usage (CPU & Memory)
- Fault Tolerance Test Results

---

## Key Functions

### 1. **Data Preprocessing**
- Normalizes the MNIST dataset.
- Converts labels into one-hot encoding.

### 2. **Distributed Training (`train_worker`)**
- Ray workers handle data splits.
- Trains a TensorFlow model on assigned splits.

### 3. **Fault Tolerance (`faulty_worker`)**
- Randomly simulates worker failures.
- Successfully retries failed workers.

### 4. **Scalability Testing**
- Varies the number of workers (e.g., 2, 4, 6, 8, 9, 12).
- Measures training time and accuracy.

### 5. **Resource Monitoring**
- Tracks CPU cores and memory usage.

---

## Fault Tolerance Test
- Simulates random worker failures.
- Measures recovery performance and final average accuracy.

### Example Output:
```
Fault Tolerance Test Passed. Avg Accuracy (successful workers): 0.9645
Total workers: 5, Failed workers: 1, Successful workers: 4
```

---

## Scalability Results
- Training and inference time improve with increased workers.
- Graphical representation of performance can be visualized via plots.

---

## Resource Monitoring Output Example
```
Resource Usage:
Total CPU Cores: 8
Available CPU Cores: 4
Total Memory: 16.0 GB
Used Memory: 8.5 GB
```

---

## Shutdown
After execution, Ray shuts down gracefully:
```python
ray.shutdown()
```

---

## License
This project is licensed under the MIT License.

---

## Contact
**Muj86218@gmail.com**

**Happy Distributed Training! 🚀**

# Pytorch 
# PyTorch Scalability and Fault Tolerance Experiment

## Overview
This project evaluates the scalability and fault tolerance of PyTorch models by simulating training on datasets of varying sizes and observing system resource usage, training time, and behavior during simulated faults.

## Description
The project includes two main experiments:

1. **Scalability Test:**
   - Measures training time and memory usage on increasing dataset sizes.
   - Plots the relationship between dataset size, training time, and memory usage.

2. **Fault Tolerance Test:**
   - Simulates a fault during model training.
   - Implements a recovery mechanism to ensure training continues seamlessly.
   - Tracks the occurrence of faults, recovery behavior, and overall resource usage.

## Requirements
- Python 3.x
- PyTorch
- Matplotlib
- psutil

Install dependencies using pip:
```bash
pip install torch matplotlib psutil
```

## Dataset
- A dummy dataset is generated with random input features (784-dimensional vectors) and random class labels (10 classes).
- Dataset sizes vary from 10,000 to 1,000,000 samples.

## Code Structure
- **DummyDataset:** Custom dataset class simulating data samples.
- **train_and_measure:** Measures training time and memory usage for different dataset sizes.
- **train_and_measure_with_fault_tolerance:** Simulates faults during training and implements recovery mechanisms.
- **Visualization:** Plots graphs for training time and memory usage.

## Usage
Run the script to perform scalability and fault tolerance experiments:
```bash
python pytorch_experiment.py
```

### Outputs
- Training time and memory usage for each dataset size.
- Fault occurrence and recovery behavior during training.
- Two plots:
   1. **Training Time vs Dataset Size**
   2. **Memory Usage vs Dataset Size**

## Example Output
```
Dataset Size: 10000, Training Time: 5.2s, Memory Used: 45.6 MB
Dataset Size: 50000, Training Time: 20.1s, Memory Used: 85.3 MB
...
Dataset Size: 1000000, Training Time: 150.7s, Memory Used: 450.2 MB
```

## Results Analysis
- **Scalability:** Observes how training time and memory scale with dataset size.
- **Fault Tolerance:** Verifies if faults are handled gracefully without halting training.

## Visualization
- **Training Time Plot:** Shows how training time scales with dataset size.
- **Memory Usage Plot:** Displays memory consumption across different dataset sizes.

## Conclusion
This project demonstrates PyTorch's ability to scale with dataset size and handle simulated faults, ensuring stable and efficient model training.

## License
This project is licensed under the MIT License.

## Contact
**muj86218@gmail.com**




