import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._

object SpamDetection {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Email Spam Detection")
      .master("local[*]")
      .getOrCreate()

    val df = spark.read.option("header", "true").csv("spam.csv")

    val data = df.select(col("v1").alias("label"), col("v2").alias("text"))
      .na.drop()
      .dropDuplicates()

    val labeled = data.withColumn("label",
      when(lower(col("label")) === "spam", 1).otherwise(0))

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val wordsData = tokenizer.transform(labeled)

    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val cleanData = remover.transform(wordsData)

    val tf = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures")
    val featurized = tf.transform(cleanData)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurized)
    val finalData = idfModel.transform(featurized)

    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val model = lr.fit(finalData)

    val predictions = model.transform(finalData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: " + accuracy)
  }
}
