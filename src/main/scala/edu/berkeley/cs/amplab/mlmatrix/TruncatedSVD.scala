package edu.berkeley.cs.amplab.mlmatrix

import java.util.concurrent.ThreadLocalRandom

import breeze.linalg._
import breeze.linalg.svd.SVD

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

class TruncatedSVD extends Logging with Serializable {
  def compute(A: RowPartitionedMatrix, k: Int, q: Int):
  (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double]) = {
    val m = A.numRows
    val n = A.numCols.toInt

    val colA = TruncatedSVD.rowToColumnPartitionedMatrix(A)

    val l = k + 5

    //Omega is a gaussian - this is maybe not worth doing distributed.
    val Omega = TruncatedSVD.drawGaussian(A.rdd.context, n, l, A.rdd.partitions.length).collect

    val Y = TruncatedSVD.times(A, Omega).collect

    var Q = svd(Y).leftVectors

    var i = 0
    while (i < q) {
      val YHat = TruncatedSVD.times(Q.t, colA).collect
      Q = svd(YHat.t).leftVectors

      val Yj = TruncatedSVD.times(A, Q).collect
      Q = svd(Yj).leftVectors

      i+=1
    }

    val B = TruncatedSVD.times(Q, colA).collect
    val ubsv = svd(B)
    val U = Q*ubsv.leftVectors

    (U, ubsv.singularValues, ubsv.rightVectors)

  }

}

object TruncatedSVD extends Logging {

  /**
   * Returns (matrix multiply) of A*B^T
   * @param A A matrix.
   * @param B Another matrix.
   * @return A*B^T
   */
  def times(A: RowPartitionedMatrix, B: DenseMatrix[Double]): RowPartitionedMatrix = {
    logDebug(s"In Row x Dense: Dimensions of A: (${A.numRows},${A.numCols}), Dimensions of B: (${B.rows},${B.cols})")
    val bB = A.rdd.context.broadcast(B)
    A.mapPartitions(a => {
      logDebug(s"In Row x Dense mapper: Dimensions of a: (${a.rows},${a.cols}), Dimensions of B: (${bB.value.rows},${bB.value.cols})")
      a*bB.value
    })
  }

  def times(A: DenseMatrix[Double], B: ColumnPartitionedMatrix): ColumnPartitionedMatrix = {
    logDebug(s"In Dense x Col: Dimensions of A: (${A.rows},${A.cols}), Dimensions of B: (${B.numRows},${B.numCols})")
    val aB = B.rdd.context.broadcast(A)
    B.mapPartitions(b => {
      logDebug(s"In Dense x Col mapper: Dimensions of B: (${aB.value.rows},${aB.value.cols}), Dimensions of a: (${b.rows},${b.cols})")
      aB.value*b
    })
  }

  //TODO - Evan: Implement this and its inverse.
  def rowToColumnPartitionedMatrix(A: RowPartitionedMatrix): ColumnPartitionedMatrix = {
    //For each row set, flatMap to little blocks with columnStart as the key, and rowNumber and block level
    //as the value.
    val cols = A.numCols.toInt
    val rows = A.numRows().toInt
    val colsPerPartition = cols / A.getPartitionInfo.size

    val colRdd = A.rdd.mapPartitionsWithIndex { case (part, iter) => {
      //Now that I have block id and partition info and partition, split this up.
      val lmat = iter.next().mat
      //Chunk up the parts.
      val parts = (0 until cols by colsPerPartition).map(s => (s, lmat(::, s until min(s + colsPerPartition, cols))))

      parts.map { case (colGroup, lm) => (colGroup, (part, lm))}.toIterator
      }
    }.groupByKey().map { case (id,block) => {
      //groupByKey, sortBy rowStart, and horzcat.
      val parts = block.toSeq.sortBy(x => x._1).map(x => x._2)
      parts.reduceLeft((a,b) => DenseMatrix.vertcat(a,b))
      }

    }

    new ColumnPartitionedMatrix(colRdd.map(mat => ColumnPartition(mat)), Some(rows), Some(cols))
  }

  /**
   * Draws a random gaussian matrix in a distributed fashion.
   * @param sc Context under which to create the matrix.
   * @param numRows Number of rows in the desired matrix.
   * @param numCols Number of columns in the desired matrix.
   * @param numParts Number of partitions in the desired matrix.
   * @return The matrix.
   */
  def drawGaussian(sc: SparkContext, numRows: Int, numCols: Int, numParts: Int): RowPartitionedMatrix = {
    val rowsPerPart = numRows / numParts
    val matrixParts = sc.parallelize(1 to numParts, numParts).mapPartitions { part =>
      val data = new Array[Double](rowsPerPart * numCols)
      var i = 0
      while (i < rowsPerPart * numCols) {
        data(i) = ThreadLocalRandom.current().nextGaussian()
        i = i + 1
      }
      val mat = new DenseMatrix[Double](rowsPerPart, numCols, data)
      Iterator(mat)
    }
    RowPartitionedMatrix.fromMatrix(matrixParts)
  }

  def main(args: Array[String]) {
    if (args.length < 6) {
      println("Usage: TruncatedSVD <master> <numRows> <numCols> <numParts> <rank> <exponent>")
      System.exit(0)
    }

    val sparkMaster = args(0)
    val numRows = args(1).toInt
    val numCols = args(2).toInt
    val numParts = args(3).toInt
    val rank = args(4).toInt
    val exponent = args(5).toInt

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("TruncatedSVD")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    val A = drawGaussian(sc, numRows, numCols, numParts)

    A.rdd.cache().count()

    val At = rowToColumnPartitionedMatrix(A)

    logInfo(s"Dimensions of A: (${A.numRows},${A.numCols}), Dimensions of At: (${At.numRows},${At.numCols})")

    var begin = System.nanoTime()

    val res = new TruncatedSVD().compute(A, rank, exponent)
    var end = System.nanoTime()
    logInfo(s"Truncated SVD of ${numRows}x${numCols} took ${(end - begin)/1e6}ms")

    println(norm(res._2))
    //logInfo(s"The norms of (U,S,V) are: ${norm(res._1)}, ${norm(res._2)}, ${norm(res._3)}")

    var c = readChar
    sc.stop()
  }
}