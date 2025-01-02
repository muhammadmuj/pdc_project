import time
import psutil
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.datasets import load_iris
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.master("local[1]").appName("IrisClassification").getOrCreate()

# Load the Iris dataset from sklearn
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Convert to Spark DataFrame
data = [(float(y[i]), float(X[i][0]), float(X[i][1]), float(X[i][2]), float(X[i][3])) for i in range(len(y))]
columns = ['label', 'feature1', 'feature2', 'feature3', 'feature4']
df = spark.createDataFrame(data, columns)

# Show the first few rows of the DataFrame
df.show(5)

# Step 1: Data Preprocessing using VectorAssembler
assembler = VectorAssembler(inputCols=['feature1', 'feature2', 'feature3', 'feature4'], outputCol='features')
df = assembler.transform(df)

# Step 2: Split the data into training and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=1234)

# Measure the start time for training
start_time = time.time()

# Step 3: Initialize and train the Logistic Regression model
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)

# Simulate fault during training (e.g., a random failure)
try:
    lr_model = lr.fit(train_data)
except Exception as e:
    print(f"Error during training: {e}")
    print("Simulating fault tolerance by recovering and retrying...")

    # Simulate fault tolerance by retrying the operation
    retry_start_time = time.time()
    lr_model = lr.fit(train_data)
    retry_training_time = time.time() - retry_start_time
    print(f"Recovery time after failure: {retry_training_time:.2f} seconds")

# Measure the training time after recovery
training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds (including fault tolerance)")

# Measure memory usage during training
process = psutil.Process()
memory_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB
print(f"Memory before training: {memory_before:.2f} MB")

# Step 4: Make predictions on the test data
start_time = time.time()
predictions = lr_model.transform(test_data)

# Measure the inference time
inference_time = time.time() - start_time
print(f"Inference time: {inference_time:.2f} seconds")

# Measure memory usage after inference
memory_after = process.memory_info().rss / (1024 * 1024)  # Memory in MB
print(f"Memory after inference: {memory_after:.2f} MB")

# Step 5: Evaluate the model
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.2f}")

# Stop the Spark session
spark.stop()

# Plot the scalability results (Training Time, Inference Time, Memory Usage, and Fault Tolerance)
labels = ['Training Time', 'Inference Time', 'Memory Before', 'Memory After', 'Recovery Time']
values = [training_time, inference_time, memory_before, memory_after, retry_training_time if 'retry_training_time' in locals() else 0]

# Create bar plot
fig, ax = plt.subplots()
ax.bar(labels, values, color=['blue', 'green', 'orange', 'red', 'purple'])

# Add labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Time (Seconds) / Memory (MB)')
ax.set_title('Scalability and Fault Tolerance Metrics of Apache Spark MLlib')

# Show plot
plt.show()
