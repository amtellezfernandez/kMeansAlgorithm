// Databricks notebook source
// MAGIC %md
// MAGIC # Large scale linear regression and gradient descendant optimization

// COMMAND ----------

// MAGIC %md
// MAGIC # Stochastic Gradient Descent (SGD) 
// MAGIC ### in Scala
// MAGIC
// MAGIC The primary objective of this code is to train a linear model using stochastic gradient descent (SGD) on a distributed dataset. The model aims to learn the relationship between input features and the target output by minimizing the mean squared error (MSE) between predicted and actual values. The learning rate is dynamically adjusted to ensure effective convergence of the model weights.
// MAGIC
// MAGIC
// MAGIC
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC We are training a simple linear regression model where the relationship between the input x and the output y is assumed to be linear. 
// MAGIC sum: Performs element-wise addition of two arrays. Used for summing gradients.
// MAGIC subtr: Performs element-wise subtraction of two arrays. Used to update the weights.
// MAGIC prodbyscal: Multiplies each element of an array by a scalar value. Used to scale gradients by the learning rate.
// MAGIC prods: Computes the dot product of two arrays. Used to calculate the predicted output in the model.
// MAGIC sigma: Computes the gradient of the loss function with respect to the weights for a single training example.

// COMMAND ----------

sc

// COMMAND ----------

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import scala.util.Random
val conf = new SparkConf().setAppName("SGDExample").setMaster("local[*]")
//val sc = new SparkContext(conf)

// COMMAND ----------

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import scala.util.Random
import breeze.linalg._

// COMMAND ----------

// MAGIC %md
// MAGIC In spark we need support functions for vector operations

// COMMAND ----------

import org.apache.spark.rdd.RDD
import scala.util.Random

// Function to perform element-wise addition of two arrays
def sum(a: Array[Double], b: Array[Double]): Array[Double] = {
  (a zip b).map { case (x, y) => x + y }
}

// Function to perform element-wise subtraction of two arrays
def subtr(a: Array[Double], b: Array[Double]): Array[Double] = {
  (a zip b).map { case (x, y) => x - y }
}

// Function to perform element-wise multiplication of an array by a scalar
def prodbyscal(scalar: Double, vector: Array[Double]): Array[Double] = {
  vector.map(x => x * scalar)
}

// Function to compute the dot product of two arrays
def prods(a: Array[Double], b: Array[Double]): Double = {
  (a zip b).map { case (x, y) => x * y }.sum
}

// Function to compute the gradient for a single data point
def sigma(y: Double, phi: Array[Double], current_w: Array[Double]): Array[Double] = {
  prodbyscal(2 * (prods(current_w, phi) - y), phi)
}


// COMMAND ----------

// MAGIC %md
// MAGIC #####To incorporate scaling into the SGD process and ensure that each component of the observation is scaled correctly -> normalize the features of the training data before applying SGD

// COMMAND ----------

// Function to standardize the dataset
def standardize(dtrain: RDD[Array[Double]]): (RDD[Array[Double]], Array[Double], Array[Double]) = {
  val n = dtrain.count().toDouble

  // Calculate the mean for each feature
  val mean = dtrain.reduce((a, b) => sum(a, b)).map(_ / n)

  // Calculate the variance for each feature
  val variance = dtrain.map(row => row.zip(mean).map { case (x, m) => math.pow(x - m, 2) })
    .reduce((a, b) => sum(a, b))
    .map(_ / n)

  // Calculate the standard deviation for each feature
  val stdDev = variance.map(math.sqrt)

  // Standardize the data
  val standardizedData = dtrain.map(row => row.zip(mean.zip(stdDev)).map {
    case (x, (m, s)) => (x - m) / s
  })

  (standardizedData, mean, stdDev)
}

// COMMAND ----------

// MAGIC %md
// MAGIC dtrain: The training dataset in the form of an RDD, where each element is a tuple containing a target value (y) and a feature vector (phi).
// MAGIC init_w: Initial weights of the model, initialized to zeros.
// MAGIC nb_of_epochs: Number of epochs to run the SGD.
// MAGIC initial_stepsize: The initial learning rate for gradient descent.
// MAGIC decay_rate: The rate at which the learning rate decays over epochs.
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC Initializes the weights w to the given init_w.
// MAGIC For each epoch:
// MAGIC The learning rate (stepsize) is updated using the decay formula.
// MAGIC The training data is partitioned, and each partition is processed in parallel.
// MAGIC Within each partition, the training examples are randomly shuffled to ensure stochasticity.
// MAGIC For each training example in the shuffled order:
// MAGIC Computes the gradient using the sigma function.
// MAGIC Updates the weights w using the computed gradient and the current learning rate.
// MAGIC The function returns the optimized weights after all epochs are completed.

// COMMAND ----------


// Function to calculate the Mean Squared Error
def computeMSE(dtrain: RDD[(Double, Array[Double])], w: Array[Double]): Double = {
  dtrain.map { case (y, phi) =>
    val prediction = prods(w, phi)
    math.pow(y - prediction, 2)
  }.mean()
}

// Function to perform SGD with a variable step size and track weights evolution
def sgd_partitions_perm_with_tracking(
    dtrain: RDD[(Double, Array[Double])],
    init_w: Array[Double],
    nb_of_epochs: Int,
    initial_stepsize: Double,
    decay_rate: Double
): (Array[Double], List[(Double, Double)], List[Double]) = {
  
  var w = init_w
  var weightHistory = List[(Double, Double)]() // List to store slope and intercept at each epoch
  var lossHistory = List[Double]() // List to store loss at each epoch

  for (epoch <- 1 to nb_of_epochs) {
    // Calculate the step size for the current epoch
    val stepsize = initial_stepsize / (1 + decay_rate * epoch)

    // Get the RDD of partitioned data with index
    val indexedPartitions = dtrain.glom().zipWithIndex()

    indexedPartitions.collect().foreach { case (partition, index) =>
      // Generate a random permutation of indices for the current partition
      val permutedIndices = Random.shuffle(partition.indices.toList)

      // Iterate through each training example in the permuted order
      permutedIndices.foreach { i =>
        val (y, phi) = partition(i)

        // Compute the gradient for the current point
        val gradient = sigma(y, phi, w)

        // Update weights with the variable step size
        w = subtr(w, prodbyscal(stepsize, gradient))
      }
    }

    // Track the evolution of the weights (slope and intercept)
    weightHistory = weightHistory :+ (w(0), w(1))

    // Compute and track the loss (MSE)
    val currentLoss = computeMSE(dtrain, w)
    lossHistory = lossHistory :+ currentLoss
  }

  (w, weightHistory, lossHistory)
}

// Example dataset creation
val X = (1 to 2000).by(2).toArray
val dtrain = sc.parallelize(X.map(x => ((1 * x + 2).toDouble, Array(x.toDouble, 1.0))), numSlices = 10)

// Initial parameters
val sizefeat = 2
val winit = Array.fill(sizefeat)(0.0)
val initial_stepsize = 0.00000001
val decay_rate = 0.01 // Decay rate for reducing step size
val nb_of_epochs = 2000

// Run SGD with variable step size and track weights
val (w1, weightHistory, lossHistory) = sgd_partitions_perm_with_tracking(dtrain, winit, nb_of_epochs, initial_stepsize, decay_rate)

// Print final weights
println(s"Final Weights: ${w1.mkString(", ")}")

import spark.implicits._
val weightHistoryDF = weightHistory.toDF("slope", "intercept")
val lossHistoryDF = lossHistory.toDF("loss")

// Convert DataFrame to Pandas DataFrame to use directly in Python
weightHistoryDF.createOrReplaceTempView("weightHistoryView")
lossHistoryDF.createOrReplaceTempView("lossHistoryView")


// COMMAND ----------

// MAGIC %md
// MAGIC The learning rate decay ensures that the training process becomes more stable over time, preventing oscillations around the optimal solution. are close to the expected values (1 for the slope and a small number for the intercept). This indicates that the model has effectively learned the linear relationship between x and y in the training data
// MAGIC

// COMMAND ----------

// MAGIC %md
// MAGIC The final weights w1 =1.0012261771666224, w0= 0.003966686752650602 are close to the expected values (1 for the slope and a small number for the intercept). This indicates that the model has effectively learned the linear relationship between x and y in the training data.

// COMMAND ----------

// MAGIC %python
// MAGIC # Inspect the DataFrame to ensure correct data
// MAGIC print(weight_history_df.head())
// MAGIC print(loss_history_df.head())
// MAGIC

// COMMAND ----------

// MAGIC %python
// MAGIC import pandas as pd
// MAGIC import matplotlib.pyplot as plt
// MAGIC
// MAGIC # Convert the Spark SQL temporary views to Pandas DataFrames
// MAGIC weight_history_df = spark.sql("SELECT * FROM weightHistoryView").toPandas()
// MAGIC loss_history_df = spark.sql("SELECT * FROM lossHistoryView").toPandas()
// MAGIC
// MAGIC # Plot the evolution of slope, intercept, and loss over epochs
// MAGIC plt.figure(figsize=(18, 6))
// MAGIC
// MAGIC # Plot for slope
// MAGIC plt.subplot(1, 3, 1)
// MAGIC plt.plot(weight_history_df.index, weight_history_df['slope'], label='Slope', color='blue')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Slope')
// MAGIC plt.title('Evolution of Slope over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Plot for intercept
// MAGIC plt.subplot(1, 3, 2)
// MAGIC plt.plot(weight_history_df.index, weight_history_df['intercept'], label='Intercept', color='orange')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Intercept')
// MAGIC plt.title('Evolution of Intercept over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Plot for loss
// MAGIC plt.subplot(1, 3, 3)
// MAGIC plt.plot(loss_history_df.index, loss_history_df['loss'], label='Loss', color='red')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Loss (MSE)')
// MAGIC plt.title('Loss (MSE) over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Show the plots
// MAGIC plt.tight_layout()
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %md
// MAGIC # Stochastic Gradient Descent (SGD) MOMENTUM
// MAGIC ### in Scala

// COMMAND ----------

// MAGIC %md
// MAGIC Momentum heavy ball method" is an optimization technique used to speed up the convergence of SGD and to reduce oscillations, especially in situations where the gradient direction is steep. By incorporating a momentum term, the updates to the model's parameters are smoothed, which helps the optimizer navigate the loss landscape more effectively.
// MAGIC
// MAGIC Momentum works by adding a fraction γ 
// MAGIC of the update vector of the previous iteration to the current update vector.
// MAGIC This effectively means the update at each step not only depends on the current gradient but also on the accumulation of past gradients.

// COMMAND ----------

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector, sum => bsum, *}
import breeze.numerics._

// Function to perform element-wise addition of two vectors using Breeze
def sum(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] = a + b

// Function to perform element-wise subtraction of two vectors using Breeze
def subtr(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] = a - b

// Function to perform element-wise multiplication of a vector by a scalar using Breeze
def prodbyscal(scalar: Double, vector: DenseVector[Double]): DenseVector[Double] = vector * scalar

// Function to compute the dot product of two vectors using Breeze
def prods(a: DenseVector[Double], b: DenseVector[Double]): Double = a dot b

// Function to compute the gradient for a single data point
def sigma(y: Double, phi: DenseVector[Double], current_w: DenseVector[Double]): DenseVector[Double] = {
  prodbyscal(2 * (prods(current_w, phi) - y), phi)
}

// Function to calculate the Mean Squared Error
def computeMSE(dtrain: RDD[(Double, DenseVector[Double])], w: DenseVector[Double]): Double = {
  dtrain.map { case (y, phi) =>
    val prediction = prods(w, phi)
    math.pow(y - prediction, 2)
  }.mean()
}

// Function to perform Momentum Heavy Ball method and track weights and velocity evolution
def heavy_ball_momentum(
    dtrain: RDD[(Double, DenseVector[Double])],
    init_w: DenseVector[Double],
    nb_of_epochs: Int,
    initial_stepsize: Double,
    decay_rate: Double,
    momentum: Double
): (DenseVector[Double], List[(Double, Double)], List[Double], List[DenseVector[Double]]) = {

  var w = init_w
  var velocity = DenseVector.zeros[Double](init_w.length) // Initialize velocity for momentum
  var weightHistory = List[(Double, Double)]() // List to store slope and intercept at each epoch
  var lossHistory = List[Double]() // List to store loss at each epoch
  var velocityHistory = List[DenseVector[Double]]() // List to store velocity at each epoch

  for (epoch <- 1 to nb_of_epochs) {
    println(s"Starting epoch $epoch / $nb_of_epochs") // Print statement for epoch tracking

    // Broadcast the current weights to all nodes
    val bcW = dtrain.sparkContext.broadcast(w)

    // Calculate the step size for the current epoch
    val stepsize = initial_stepsize / (1 + decay_rate * epoch)

    // Update weights for each partition using mapPartitions
    val (updatedW, updatedVelocity) = dtrain.mapPartitions { partition =>
      var localW = bcW.value.copy
      var localVelocity = velocity.copy

      partition.foreach { case (y, phi) =>
        // Compute gradient at the current position
        val gradient = sigma(y, phi, localW)

        // Update velocity with gradient and momentum
        localVelocity = sum(prodbyscal(momentum, localVelocity), prodbyscal(stepsize, gradient))

        // Update weights with the new velocity
        localW = subtr(localW, localVelocity)
      }
      Iterator((localW, localVelocity)) // Return both weights and velocity
    }.treeReduce { (a, b) =>
      (sum(a._1, b._1), sum(a._2, b._2)) // Sum both weights and velocities from all partitions
    }

    // Update the velocity
    velocity = updatedVelocity.map(_ / dtrain.getNumPartitions.toDouble)

    // Update the weights
    w = updatedW.map(_ / dtrain.getNumPartitions.toDouble)

    // Track the evolution of the weights (slope and intercept)
    weightHistory = weightHistory :+ (w(0), w(1))

    // Compute and track the loss (MSE)
    val currentLoss = computeMSE(dtrain, w)
    lossHistory = lossHistory :+ currentLoss

    // Track the velocity at the end of the epoch
    velocityHistory = velocityHistory :+ velocity.copy

    println(s"End of epoch $epoch / $nb_of_epochs - Loss: $currentLoss") // Print statement for epoch completion and loss

    // Clean up the broadcast variable to free up memory
    bcW.destroy()
  }

  (w, weightHistory, lossHistory, velocityHistory)
}

// Example dataset creation using Breeze Vectors
val X = (1 to 2000).by(2).toArray
val dtrain = sc.parallelize(X.map(x => ((1 * x + 2).toDouble, DenseVector(x.toDouble, 1.0))), numSlices = 10)

// Initial parameters
val sizefeat = 2
val winit = DenseVector.zeros[Double](sizefeat)
val initial_stepsize = 0.0000001
val decay_rate = 0.01 // Decay rate for reducing step size
val nb_of_epochs = 50
val momentum = 0.6 // Momentum coefficient

// Run Momentum Heavy Ball method and track weights and velocity
val (w1, weightHistory, lossHistory, velocityHistory) = heavy_ball_momentum(dtrain, winit, nb_of_epochs, initial_stepsize, decay_rate, momentum)

// Print final weights
println(s"Final Weights: ${w1}")

// Convert weight history and velocity history to DataFrames
import spark.implicits._
val weightHistoryDF = weightHistory.toDF("slope", "intercept")
val lossHistoryDF = lossHistory.toDF("loss")
val velocityHistoryDF = velocityHistory.map(v => (v(0), v(1))).toDF("velocity_slope", "velocity_intercept")

// Register as temporary views for use in SQL or conversion to Pandas
weightHistoryDF.createOrReplaceTempView("weightHistoryView")
lossHistoryDF.createOrReplaceTempView("lossHistoryView")
velocityHistoryDF.createOrReplaceTempView("velocityHistoryView")


// COMMAND ----------

// MAGIC %python
// MAGIC import pandas as pd
// MAGIC import matplotlib.pyplot as plt
// MAGIC
// MAGIC # Convert the Spark SQL temporary views to Pandas DataFrames
// MAGIC weight_history_df = spark.sql("SELECT * FROM weightHistoryView").toPandas()
// MAGIC loss_history_df = spark.sql("SELECT * FROM lossHistoryView").toPandas()
// MAGIC
// MAGIC # Plot the evolution of slope, intercept, and loss over epochs
// MAGIC plt.figure(figsize=(18, 6))
// MAGIC
// MAGIC # Plot for slope
// MAGIC plt.subplot(1, 3, 1)
// MAGIC plt.plot(weight_history_df.index, weight_history_df['slope'], label='Slope', color='blue')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Slope')
// MAGIC plt.title('Evolution of Slope over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Plot for intercept
// MAGIC plt.subplot(1, 3, 2)
// MAGIC plt.plot(weight_history_df.index, weight_history_df['intercept'], label='Intercept', color='orange')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Intercept')
// MAGIC plt.title('Evolution of Intercept over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Plot for loss
// MAGIC plt.subplot(1, 3, 3)
// MAGIC plt.plot(loss_history_df.index, loss_history_df['loss'], label='Loss', color='red')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Loss (MSE)')
// MAGIC plt.title('Loss (MSE) over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Show the plots
// MAGIC plt.tight_layout()
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC
// MAGIC # Retrieve the DataFrames from Spark SQL temporary views
// MAGIC velocity_df = spark.sql("SELECT * FROM velocityHistoryView").toPandas()
// MAGIC
// MAGIC # Plot the velocity evolution
// MAGIC plt.figure(figsize=(10, 6))
// MAGIC plt.plot(velocity_df.index, velocity_df['velocity_slope'], label='Velocity Slope')
// MAGIC plt.plot(velocity_df.index, velocity_df['velocity_intercept'], label='Velocity Intercept')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Velocity')
// MAGIC plt.title('Velocity Evolution Over Epochs')
// MAGIC plt.legend()
// MAGIC plt.grid(True)
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %md
// MAGIC Convergence with the same parameters is much more faster in comparison? HOWEVER, covergence is very fast, this rapid convergence can sometimes lead to issues, as we see/
// MAGIC
// MAGIC If the model converges too quickly, especially on noisy or small datasets, it may fit too closely to the training data and not generalize well to new data.
// MAGIC
// MAGIC The increased speed can cause the model to overshoot the optimal point, especially if the learning rate is not carefully controlled. This can result in the model bouncing around the optimal parameters without settling into the best solution.

// COMMAND ----------

// MAGIC %md
// MAGIC # Nesterov accelerated gradient
// MAGIC NAG takes this a step further. Before deciding on the next step, NAG allows us to "peek" at where the current momentum is about to take us. Instead of blindly following the current path, it checks ahead, estimates the gradient (or slope) at this new position, and adjusts the course accordingly. This way, the optimization process becomes smarter—it anticipates where it is heading and adjusts its path more efficiently, reducing the likelihood of overshooting and speeding up convergence towards the optimal solution.
// MAGIC
// MAGIC

// COMMAND ----------

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector, sum => bsum, *}
import breeze.numerics._

// Function to perform element-wise addition of two vectors using Breeze
def sum(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] = a + b

// Function to perform element-wise subtraction of two vectors using Breeze
def subtr(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] = a - b

// Function to perform element-wise multiplication of a vector by a scalar using Breeze
def prodbyscal(scalar: Double, vector: DenseVector[Double]): DenseVector[Double] = vector * scalar

// Function to compute the dot product of two vectors using Breeze
def prods(a: DenseVector[Double], b: DenseVector[Double]): Double = a dot b

// Function to compute the gradient for a single data point
def sigma(y: Double, phi: DenseVector[Double], current_w: DenseVector[Double]): DenseVector[Double] = {
  prodbyscal(2 * (prods(current_w, phi) - y), phi)
}

// Function to calculate the Mean Squared Error
def computeMSE(dtrain: RDD[(Double, DenseVector[Double])], w: DenseVector[Double]): Double = {
  dtrain.map { case (y, phi) =>
    val prediction = prods(w, phi)
    math.pow(y - prediction, 2)
  }.mean()
}

// Function to perform Nesterov Accelerated Gradient (NAG) with momentum and track weights and velocity evolution
def nag_with_momentum(
    dtrain: RDD[(Double, DenseVector[Double])],
    init_w: DenseVector[Double],
    nb_of_epochs: Int,
    initial_stepsize: Double,
    decay_rate: Double,
    momentum: Double
): (DenseVector[Double], List[(Double, Double)], List[Double], List[DenseVector[Double]]) = {

  var w = init_w
  var velocity = DenseVector.zeros[Double](init_w.length) // Initialize velocity for momentum
  var weightHistory = List[(Double, Double)]() // List to store slope and intercept at each epoch
  var lossHistory = List[Double]() // List to store loss at each epoch
  var velocityHistory = List[DenseVector[Double]]() // List to store velocity at each epoch

  for (epoch <- 1 to nb_of_epochs) {
    println(s"Starting epoch $epoch / $nb_of_epochs") // Print statement for epoch tracking

    // Calculate the step size for the current epoch
    val stepsize = initial_stepsize / (1 + decay_rate * epoch)

    // Perform look-ahead step by applying the current velocity to weights
    val lookAheadW = subtr(w, prodbyscal(momentum, velocity))

    // Broadcast the look-ahead weights to all nodes
    val bcLookAheadW = dtrain.sparkContext.broadcast(lookAheadW)

    // Update weights for each partition using mapPartitions
    val (updatedW, updatedVelocity) = dtrain.mapPartitions { partition =>
      var localW = bcLookAheadW.value.copy
      var localVelocity = velocity.copy

      partition.foreach { case (y, phi) =>
        // Compute gradient at the look-ahead position
        val gradient = sigma(y, phi, localW)

        // Update velocity with gradient and momentum
        localVelocity = sum(prodbyscal(momentum, localVelocity), prodbyscal(stepsize, gradient))
      }

      // Update weights with the new velocity after aggregating all partitions
      Iterator((subtr(w, localVelocity), localVelocity)) // Return both weights and velocity
    }.treeReduce { (a, b) =>
      (sum(a._1, b._1), sum(a._2, b._2)) // Sum both weights and velocities from all partitions
    }

    // Update the velocity
    velocity = updatedVelocity.map(_ / dtrain.getNumPartitions.toDouble)

    // Update the weights
    w = updatedW.map(_ / dtrain.getNumPartitions.toDouble)

    // Track the evolution of the weights (slope and intercept)
    weightHistory = weightHistory :+ (w(0), w(1))

    // Compute and track the loss (MSE)
    val currentLoss = computeMSE(dtrain, w)
    lossHistory = lossHistory :+ currentLoss

    // Track the velocity at the end of the epoch
    velocityHistory = velocityHistory :+ velocity.copy

    println(s"End of epoch $epoch / $nb_of_epochs - Loss: $currentLoss") // Print statement for epoch completion and loss

    // Clean up the broadcast variable to free up memory
    bcLookAheadW.destroy()
  }

  (w, weightHistory, lossHistory, velocityHistory)
}

// Example dataset creation using Breeze Vectors
val X = (1 to 2000).by(2).toArray
val dtrain = sc.parallelize(X.map(x => ((1 * x + 2).toDouble, DenseVector(x.toDouble, 1.0))), numSlices = 10)

// Initial parameters
val sizefeat = 2
val winit = DenseVector.zeros[Double](sizefeat)
val initial_stepsize = 0.0000001
val decay_rate = 0.01 // Decay rate for reducing step size
val nb_of_epochs = 50
val momentum = 0.6 // Momentum coefficient

// Run Nesterov Accelerated Gradient (NAG) with momentum and track weights and velocity
val (w1, weightHistory, lossHistory, velocityHistory) = nag_with_momentum(dtrain, winit, nb_of_epochs, initial_stepsize, decay_rate, momentum)

// Print final weights
println(s"Final Weights: ${w1}")

// Convert weight history and velocity history to DataFrames
import spark.implicits._
val weightHistoryDF = weightHistory.toDF("slope", "intercept")
val lossHistoryDF = lossHistory.toDF("loss")
val velocityHistoryDF = velocityHistory.map(v => (v(0), v(1))).toDF("velocity_slope", "velocity_intercept")

// Register as temporary views for use in SQL or conversion to Pandas
weightHistoryDF.createOrReplaceTempView("weightHistoryView")
lossHistoryDF.createOrReplaceTempView("lossHistoryView")
velocityHistoryDF.createOrReplaceTempView("velocityHistoryView")


// COMMAND ----------

// MAGIC %python
// MAGIC import pandas as pd
// MAGIC import matplotlib.pyplot as plt
// MAGIC
// MAGIC # Convert the Spark SQL temporary views to Pandas DataFrames
// MAGIC weight_history_df = spark.sql("SELECT * FROM weightHistoryView").toPandas()
// MAGIC loss_history_df = spark.sql("SELECT * FROM lossHistoryView").toPandas()
// MAGIC
// MAGIC # Plot the evolution of slope, intercept, and loss over epochs
// MAGIC plt.figure(figsize=(18, 6))
// MAGIC
// MAGIC # Plot for slope
// MAGIC plt.subplot(1, 3, 1)
// MAGIC plt.plot(weight_history_df.index, weight_history_df['slope'], label='Slope', color='blue')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Slope')
// MAGIC plt.title('Evolution of Slope over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Plot for intercept
// MAGIC plt.subplot(1, 3, 2)
// MAGIC plt.plot(weight_history_df.index, weight_history_df['intercept'], label='Intercept', color='orange')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Intercept')
// MAGIC plt.title('Evolution of Intercept over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Plot for loss
// MAGIC plt.subplot(1, 3, 3)
// MAGIC plt.plot(loss_history_df.index, loss_history_df['loss'], label='Loss', color='red')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Loss (MSE)')
// MAGIC plt.title('Loss (MSE) over Epochs')
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Show the plots
// MAGIC plt.tight_layout()
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC
// MAGIC # Retrieve the DataFrames from Spark SQL temporary views
// MAGIC velocity_df = spark.sql("SELECT * FROM velocityHistoryView").toPandas()
// MAGIC
// MAGIC # Plot the velocity evolution
// MAGIC plt.figure(figsize=(10, 6))
// MAGIC plt.plot(velocity_df.index, velocity_df['velocity_slope'], label='Velocity Slope')
// MAGIC plt.plot(velocity_df.index, velocity_df['velocity_intercept'], label='Velocity Intercept')
// MAGIC plt.xlabel('Epoch')
// MAGIC plt.ylabel('Velocity')
// MAGIC plt.title('Velocity Evolution Over Epochs')
// MAGIC plt.legend()
// MAGIC plt.grid(True)
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %md
// MAGIC The different behaviors between the two optimization methods—the Momentum Heavy Ball method and the Nesterov Accelerated Gradient—can be understood by looking at how each method works and how they respond to the data during training.
// MAGIC
// MAGIC In the first set of plots, where we see a smoother and faster convergence, this represents the Momentum Heavy Ball method. This method uses both the gradient of the current step and a "momentum" term, which is like adding a little push in the direction it’s already moving. This momentum helps to keep the optimization moving in a consistent direction, which reduces any back-and-forth movements (oscillations) and speeds up the process of finding the minimum. The result is a smoother path toward the minimum because the method builds up speed when moving in the right direction and doesn't change direction abruptly. That’s why the line in the plot moves steadily toward the goal without much wobbling.
// MAGIC
// MAGIC The second set of plots shows a more wiggly or oscillatory path, which represents the Nesterov Accelerated Gradient method. Nesterov’s method is a bit more proactive or "look-ahead" in its approach. Instead of just following the current gradient, it first takes a step in the direction of the momentum, then calculates the gradient as if it’s already a bit further along the path. This look-ahead step makes Nesterov’s method more dynamic, which can lead to more oscillations as it adjusts its direction more frequently.
// MAGIC
// MAGIC Even though these oscillations might seem like a bad thing at first because they look like the method is "bouncing" around a lot, they can actually be helpful. The oscillations show that Nesterov’s method is actively adjusting its path and being careful not to overshoot the minimum. This means that even though it wiggles a bit more, it can help the optimization to be more precise and possibly reach the minimum faster. In some cases, this makes Nesterov’s method more effective than the Momentum Heavy Ball method, especially when navigating complex or bumpy landscapes where the minimum isn’t straightforward.
// MAGIC
// MAGIC So, while the Momentum Heavy Ball method gives a smoother and more straightforward path to the goal, the oscillations seen in Nesterov’s method are actually a positive sign that the method is being more careful and adaptive, potentially leading to better results in certain situations.

// COMMAND ----------


