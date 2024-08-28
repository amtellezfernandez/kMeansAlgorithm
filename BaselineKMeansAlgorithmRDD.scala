// Databricks notebook source
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import scala.math.pow
import scala.util.Random
import spark.implicits._

// COMMAND ----------

import org.apache.spark.sql.SparkSession

// Initialize SparkSession
val spark = SparkSession.builder()
  .appName("KMeansFromScratch")
  .master("local[*]")
  .config("spark.driver.memory", "4g") 
  .config("spark.executor.memory", "4g") 
  .getOrCreate()



// COMMAND ----------

// Set the legacy time parser policy
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")

// Define the schema for the CSV file
val schema = StructType(
  StructField("time", TimestampType, nullable = true) ::
  StructField("lat", DoubleType, nullable = true) ::
  StructField("lon", DoubleType, nullable = true) ::
  StructField("base", StringType, nullable = true) ::
  Nil
)

// Read the CSV file into DataFrame with defined schema
val uberDf = spark.read.format("csv")
  .option("header", "true")
  .option("delimiter", ",")
  .option("mode", "DROPMALFORMED")
  .option("timestampFormat", "M/d/yyyy H:mm:ss")
  .schema(schema)
  .load("/FileStore/tables/uber.csv")
  .cache()


uberDf.printSchema()
uberDf.show(10)

// COMMAND ----------

// Prepare data for clustering using VectorAssembler
val cols = Array("lat", "lon")
val assembler = new VectorAssembler().setInputCols(cols).setOutputCol("features")
val featureDf = assembler.transform(uberDf)

// Convert DataFrame to RDD of features
val dataRDD = featureDf.select("features").rdd.map(row => row.getAs[Vector]("features"))
val pointsRDD = dataRDD.map(v => (v(0), v(1)))

// COMMAND ----------

// Define number of clusters and convergence distance
val K = 5
val convergeDist = 0.1

// Initialize K random points as cluster centers
var kPoints = pointsRDD.takeSample(false, K, 42)
println("Initial K Center points:")
kPoints.foreach(println)

// Define utility functions for KMeans algorithm
def distanceSquared(p1: (Double, Double), p2: (Double, Double)): Double = {
  pow(p1._1 - p2._1, 2) + pow(p1._2 - p2._2, 2)
}

def addPoints(p1: (Double, Double), p2: (Double, Double)): (Double, Double) = {
  (p1._1 + p2._1, p1._2 + p2._2)
}

def closestPoint(p: (Double, Double), points: Array[(Double, Double)]): Int = {
  var bestIndex = 0
  var closest = Double.PositiveInfinity

  for (i <- points.indices) {
    val dist = distanceSquared(p, points(i))
    if (dist < closest) {
      closest = dist
      bestIndex = i
    }
  }
  bestIndex
}

// Initialize lists to store iteration times and distances
var tempDist = Double.PositiveInfinity
var iterationTimes = List[(Int, Double, Long)]() // (Iteration, Distance, Time)
var iteration = 0

// Initialize variables for additional time metrics
var samplingTime: Long = 0
var assignmentTime: Long = 0
var statisticsComputationTime: Long = 0
var interClusterDistanceComputationTime: Long = 0

// KMeans Clustering Algorithm
while (tempDist > convergeDist) {
  val startTime = System.currentTimeMillis()  // Start time

  // Map each point to the closest cluster center
  val closestToKpointRDD = pointsRDD.map(point => (closestPoint(point, kPoints), (point, 1)))

  // Reduce to find new cluster centers
  val pointCalculatedRDD = closestToKpointRDD.reduceByKey { case ((point1, n1), (point2, n2)) =>
    (addPoints(point1, point2), n1 + n2)
  }

  // Calculate new cluster centers
  val newPoints = pointCalculatedRDD.map { case (i, (point, n)) =>
    (i, (point._1 / n, point._2 / n))
  }.collectAsMap()

  // Calculate distance between old and new centers
  tempDist = 0.0
  for (i <- 0 until K) {
    tempDist += distanceSquared(kPoints(i), newPoints(i))
  }

  val endTime = System.currentTimeMillis()  // End time
  iteration += 1
  iterationTimes = iterationTimes :+ (iteration, tempDist, endTime - startTime) // Collect iteration time

  // Update cluster centers
  for (i <- 0 until K) {
    kPoints(i) = newPoints(i)
  }
}

// Convert iteration times to DataFrame
val iterationTimesDF = iterationTimes.toDF("Iteration", "Distance", "Time_ms")

// Sample points and measure time
val startSampleTime = System.currentTimeMillis()
val samplePoints = pointsRDD.takeSample(false, 10, 42)
val endSampleTime = System.currentTimeMillis()
samplingTime = endSampleTime - startSampleTime

// Assign points to nearest cluster and measure time
val startAssignTime = System.currentTimeMillis()
val sampleAssignments = samplePoints.map(point => (point, closestPoint(point, kPoints)))
val endAssignTime = System.currentTimeMillis()
assignmentTime = endAssignTime - startAssignTime

// Compute statistics and measure time
val startStatsTime = System.currentTimeMillis()
val clusterAssignmentsRDD = pointsRDD.map(point => (closestPoint(point, kPoints), point))
val clusterStatsRDD = clusterAssignmentsRDD.groupByKey().map { case (cluster, points) =>
  val numPoints = points.size
  val centroid = kPoints(cluster)
  val dispersion = points.map(point => distanceSquared(point, centroid)).sum / numPoints
  (cluster, numPoints, dispersion)
}.collect()
val endStatsTime = System.currentTimeMillis()
statisticsComputationTime = endStatsTime - startStatsTime

// Compute inter-cluster distances and measure time
val startInterClusterDistTime = System.currentTimeMillis()
val interClusterDistances = for {
  i <- 0 until K
  j <- i + 1 until K
} yield {
  val dist = math.sqrt(distanceSquared(kPoints(i), kPoints(j)))
  (i, j, dist)
}
val endInterClusterDistTime = System.currentTimeMillis()
interClusterDistanceComputationTime = endInterClusterDistTime - startInterClusterDistTime

// Convert cluster statistics to DataFrame
val clusterStatsDF = clusterStatsRDD.toSeq.toDF("Cluster", "Number_of_Points", "Dispersion")

// Convert inter-cluster distances to DataFrame
val interClusterDistancesDF = interClusterDistances.toSeq.toDF("Cluster_1", "Cluster_2", "Distance")

// Convert final cluster centers to DataFrame
val finalCentersDF = kPoints.zipWithIndex.map { case (point, idx) =>
  (idx, point._1, point._2)
}.toSeq.toDF("Cluster", "Center_X", "Center_Y")

// Convert sample assignments to DataFrame
val sampleAssignmentsDF = sampleAssignments.toSeq.toDF("Point", "Cluster")

// Combine all timing data into a single DataFrame
val timingsDF = Seq(
  ("Iteration Time (total)", iterationTimes.map(_._3).sum),
  ("Sampling Time", samplingTime),
  ("Assignment Time", assignmentTime),
  ("Statistics Computation Time", statisticsComputationTime),
  ("Inter-Cluster Distance Computation Time", interClusterDistanceComputationTime)
).toDF("Metric", "Time_ms")

// Display all DataFrames
iterationTimesDF.show()
finalCentersDF.show()
sampleAssignmentsDF.show()
clusterStatsDF.show()
interClusterDistancesDF.show()
timingsDF.show()

// COMMAND ----------


// Print final cluster centers
println("Final cluster centers:")
kPoints.foreach(println)

// Prepare data for plotting and ensure the correct number of columns
val clusterAssignments = pointsRDD.map(point => (closestPoint(point, kPoints), point._1, point._2)).collect()

// Convert to DataFrames for Python interoperability
val clusterAssignmentsDF = clusterAssignments.toSeq.toDF("cluster", "x", "y")
val centroidsDF = kPoints.toSeq.toDF("x", "y")

// Register DataFrames as temporary views for SQL usage or to read in Python
clusterAssignmentsDF.createOrReplaceTempView("cluster_assignments")
centroidsDF.createOrReplaceTempView("centroids")

// COMMAND ----------

// MAGIC %python
// MAGIC # Import necessary libraries
// MAGIC import matplotlib.pyplot as plt
// MAGIC import pandas as pd
// MAGIC
// MAGIC # Read data from the temporary views created in Scala
// MAGIC cluster_assignments = spark.sql("SELECT * FROM cluster_assignments").toPandas()
// MAGIC centroids = spark.sql("SELECT * FROM centroids").toPandas()
// MAGIC
// MAGIC # Prepare data for plotting
// MAGIC plt.figure(figsize=(10, 6))
// MAGIC
// MAGIC # Define a color map for clusters
// MAGIC colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
// MAGIC
// MAGIC # Plot each cluster's points
// MAGIC for cluster_id in cluster_assignments['cluster'].unique():
// MAGIC     cluster_data = cluster_assignments[cluster_assignments['cluster'] == cluster_id]
// MAGIC     x = cluster_data['x']
// MAGIC     y = cluster_data['y']
// MAGIC     plt.scatter(x, y, c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}', alpha=0.6)
// MAGIC
// MAGIC # Plot the centroids
// MAGIC plt.scatter(centroids['x'], centroids['y'], c='black', marker='X', s=100, label='Centroids')
// MAGIC
// MAGIC # Set plot title and labels
// MAGIC plt.title('Cluster Plot')
// MAGIC plt.xlabel('Feature 1')
// MAGIC plt.ylabel('Feature 2')
// MAGIC plt.legend()
// MAGIC plt.grid(True)
// MAGIC
// MAGIC # Show plot
// MAGIC plt.show()
// MAGIC

// COMMAND ----------


