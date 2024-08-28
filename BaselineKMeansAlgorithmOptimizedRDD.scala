// Databricks notebook source
import org.apache.spark.sql.SparkSession

// Initialize SparkSession
val spark = SparkSession.builder()
  .appName("KMeansFromScratch")
  .master("local[*]")
  .config("spark.driver.memory", "4g") 
  .config("spark.executor.memory", "4g") 
  .getOrCreate()


// COMMAND ----------


import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import scala.math.pow
import scala.util.Random
import spark.implicits._

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

import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector, DenseVector}
import scala.math.pow
import org.apache.spark.sql.SparkSession

// Start Spark session
val spark = SparkSession.builder
  .appName("KMeansPerformanceOptimized")
  .getOrCreate()
import spark.implicits._
val sc = spark.sparkContext

// Convert DataFrame to RDD of features (using SparseVector for memory efficiency if dataset is sparse)
val dataRDD = featureDf.select("features").rdd.map(row => row.getAs[Vector]("features"))
val pointsRDD = dataRDD.map {
  case v: DenseVector => (v(0), v(1))
  case v: SparseVector => (v.toDense(0), v.toDense(1)) // Convert sparse vectors to dense for computation
}.cache()

// Number of clusters
val K = 5
// Convergence distance
val convergeDist = 0.1

// Initialize K random points as cluster centers
var kPoints = pointsRDD.takeSample(false, K, 42)
println("Initial K Center points:")
kPoints.foreach(println)

// Function to compute squared distance between two points
def distanceSquared(p1: (Double, Double), p2: (Double, Double)): Double = {
  pow(p1._1 - p2._1, 2) + pow(p1._2 - p2._2, 2)
}

// Function to add two points
def addPoints(p1: (Double, Double), p2: (Double, Double)): (Double, Double) = {
  (p1._1 + p2._1, p1._2 + p2._2)
}

// Function to find the closest center for a given point
def closestPoint(p: (Double, Double), points: Array[(Double, Double)]): Int = {
  points.indices.minBy(i => distanceSquared(p, points(i)))
}

var tempDist = Double.PositiveInfinity

// Measure time for each iteration
while (tempDist > convergeDist) {
  val startTime = System.currentTimeMillis()  // Start time

  // Broadcast the cluster centers to all nodes to minimize data transfer
  val bcKPoints = sc.broadcast(kPoints)

  // Mapping each point to the closest cluster center using vectorized operations
  val closestToKpointRDD = pointsRDD.mapPartitions { iter =>
    val centers = bcKPoints.value
    iter.map(point => (closestPoint(point, centers), (point, 1)))
  }

  // Reducing to find new cluster centers using treeAggregate for efficient aggregation
  val pointCalculatedRDD = closestToKpointRDD.treeAggregate(Map[Int, (Double, Double, Int)]())(
    seqOp = (acc, value) => {
      val clusterIndex = value._1
      val point = value._2._1
      val clusterData = acc.getOrElse(clusterIndex, (0.0, 0.0, 0))
      acc + (clusterIndex -> (clusterData._1 + point._1, clusterData._2 + point._2, clusterData._3 + 1))
    },
    combOp = (map1, map2) => {
      (map1.keys ++ map2.keys).map { key =>
        val (sumX1, sumY1, count1) = map1.getOrElse(key, (0.0, 0.0, 0))
        val (sumX2, sumY2, count2) = map2.getOrElse(key, (0.0, 0.0, 0))
        key -> (sumX1 + sumX2, sumY1 + sumY2, count1 + count2)
      }.toMap
    }
  )

  // Calculating new cluster centers
  val newPoints = pointCalculatedRDD.mapValues { case (sumX, sumY, count) =>
    (sumX / count, sumY / count)
  }

  // Calculate the total movement of cluster centers
  tempDist = (0 until K).map(i => distanceSquared(kPoints(i), newPoints(i))).sum

  val endTime = System.currentTimeMillis()  // End time
  println(s"Distance between iterations: $tempDist")
  println(s"Iteration time: ${endTime - startTime} ms")  // Print iteration time

  // Update cluster centers
  kPoints = (0 until K).map(i => newPoints(i)).toArray

  // Unpersist broadcast variable to free up memory
  bcKPoints.unpersist()
}

// Print final cluster centers
println("Final cluster centers:")
kPoints.foreach(println)

// Sampling points and measure time
val startSampleTime = System.currentTimeMillis()  // Start time for sampling
val samplePoints = pointsRDD.takeSample(false, 10, 42)
val endSampleTime = System.currentTimeMillis()  // End time for sampling
val samplingTime = endSampleTime - startSampleTime
println(s"Sampling time: ${samplingTime} ms")

// Assign points to nearest cluster and measure time
val startAssignTime = System.currentTimeMillis()  // Start time for assignment
val sampleAssignments = samplePoints.map(point => (point, closestPoint(point, kPoints)))
val endAssignTime = System.currentTimeMillis()  // End time for assignment
val assignmentTime = endAssignTime - startAssignTime
println(s"Assignment time: ${assignmentTime} ms")

// Convert sample assignments to DataFrame
val sampleAssignmentsDF = sampleAssignments.toSeq.toDF("Point", "Cluster")

// Compute statistics using treeAggregate for efficient aggregation
val startStatsTime = System.currentTimeMillis()  // Start time for stats computation
val clusterAssignmentsRDD = pointsRDD.map(point => (closestPoint(point, kPoints), point))

val clusterSumRDD = clusterAssignmentsRDD.treeAggregate(Map[Int, (Double, Double, Int)]())(
  (acc, value) => {
    val clusterIndex = value._1
    val point = value._2
    val clusterData = acc.getOrElse(clusterIndex, (0.0, 0.0, 0))
    acc + (clusterIndex -> (clusterData._1 + point._1, clusterData._2 + point._2, clusterData._3 + 1))
  },
  (map1, map2) => {
    (map1.keys ++ map2.keys).map { key =>
      val (sumX1, sumY1, count1) = map1.getOrElse(key, (0.0, 0.0, 0))
      val (sumX2, sumY2, count2) = map2.getOrElse(key, (0.0, 0.0, 0))
      key -> (sumX1 + sumX2, sumY1 + sumY2, count1 + count2)
    }.toMap
  }
)

// Collect cluster sums and count to calculate centroids
val clusterStats = clusterSumRDD

// Compute dispersion separately using the centroids
val dispersions = clusterAssignmentsRDD.map { case (cluster, point) =>
  val (sumX, sumY, count) = clusterStats(cluster)
  val centroid = (sumX / count, sumY / count)
  (cluster, distanceSquared(point, centroid))
}.treeAggregate(Map[Int, Double]())(
  (acc, value) => {
    val (cluster, sumOfSquares) = value
    acc + (cluster -> (acc.getOrElse(cluster, 0.0) + sumOfSquares))
  },
  (map1, map2) => {
    (map1.keys ++ map2.keys).map { key =>
      key -> (map1.getOrElse(key, 0.0) + map2.getOrElse(key, 0.0))
    }.toMap
  }
).map { case (cluster, sumOfSquares) =>
  val (_, _, count) = clusterStats(cluster)
  (cluster, sumOfSquares / count)
}

val endStatsTime = System.currentTimeMillis()  // End time for stats computation
val statisticsComputationTime = endStatsTime - startStatsTime
println(s"Statistics computation time: ${statisticsComputationTime} ms")

println("\nCluster statistics:")
clusterStats.foreach { case (cluster, (sumX, sumY, count)) =>
  println(s"Cluster $cluster: Number of points = $count")
}
dispersions.foreach { case (cluster, dispersion) =>
  println(s"Cluster $cluster: Dispersion = $dispersion")
}

// Convert cluster statistics to DataFrame
val clusterStatsDF = clusterStats.toSeq.map { case (cluster, (sumX, sumY, count)) =>
  (cluster, count, sumX / count, sumY / count)
}.toDF("Cluster", "Number_of_Points", "Centroid_X", "Centroid_Y")

// Convert dispersions to DataFrame
val dispersionsDF = dispersions.toSeq.toDF("Cluster", "Dispersion")

// Compute inter-cluster distances
val startInterClusterDistTime = System.currentTimeMillis()  // Start time for inter-cluster distances
val interClusterDistances = for {
  i <- 0 until K
  j <- i + 1 until K
} yield {
  val dist = math.sqrt(distanceSquared(kPoints(i), kPoints(j)))
  (i, j, dist)
}
val endInterClusterDistTime = System.currentTimeMillis()  // End time for inter-cluster distances
val interClusterDistanceComputationTime = endInterClusterDistTime - startInterClusterDistTime
println(s"Inter-cluster distance computation time: ${interClusterDistanceComputationTime} ms")

println("\nInter-cluster distances:")
interClusterDistances.foreach { case (i, j, dist) =>
  println(s"Distance between cluster $i and cluster $j = $dist")
}

// Convert inter-cluster distances to DataFrame
val interClusterDistancesDF = interClusterDistances.toSeq.toDF("Cluster_1", "Cluster_2", "Distance")

// Convert final cluster centers to DataFrame
val finalCentersDF = kPoints.zipWithIndex.map { case (point, idx) =>
  (idx, point._1, point._2)
}.toSeq.toDF("Cluster", "Center_X", "Center_Y")

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
dispersionsDF.show()
interClusterDistancesDF.show()
timingsDF.show()


// COMMAND ----------

import spark.implicits._
// Create DataFrame for cluster assignments from the RDD
val clusterAssignmentsDF = clusterAssignmentsRDD.map {
  case (cluster, (x, y)) => (x, y, cluster)
}.toDF("x", "y", "cluster") // This should work since it's already an RDD

// Create DataFrame for centroids
val centroidsDF = kPoints.zipWithIndex.map {
  case ((x, y), cluster) => (cluster, x, y)
}.toSeq.toDF("cluster", "x", "y") // Convert Array to Seq first, then toDF

// Register DataFrames as temporary views
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


