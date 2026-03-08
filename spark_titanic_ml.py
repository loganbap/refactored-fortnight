from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def main():
    # 1. Spark session with Hive support
    spark = (
        SparkSession.builder
        .appName("TitanicSurvivalLogisticRegression")
        .enableHiveSupport()
        .getOrCreate()
    )

    # 2. Load data from Hive
    df = spark.sql("""
        SELECT
            Survived,
            Pclass,
            Sex,
            Age,
            SibSp,
            Parch,
            Fare
        FROM titanic_hive
    """)
    print("Initial row count:", df.count())

    # 3. Basic cleaning: drop rows with nulls in label or features
    cols_needed = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    df_clean = df.dropna(subset=cols_needed)
    print("Row count after dropping nulls:", df_clean.count())

    # Ensure label is integer/double and features are numeric
    df_clean = df_clean.select(
        col("Survived").cast("double").alias("label"),
        col("Pclass").cast("double"),
        col("Sex").cast("double"),
        col("Age").cast("double"),
        col("SibSp").cast("double"),
        col("Parch").cast("double"),
        col("Fare").cast("double")
    )

    # 4. Assemble feature vector
    feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    assembled = assembler.transform(df_clean).select("label", "features")

    # 5. Train/test split
    train_df, test_df = assembled.randomSplit([0.7, 0.3], seed=42)
    print("Train rows:", train_df.count())
    print("Test rows:", test_df.count())

    # 6. Logistic Regression model
    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        maxIter=50,
        regParam=0.0,
        elasticNetParam=0.0
    )
    lr_model = lr.fit(train_df)
    print("Logistic Regression model trained.")
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)

    # 7. Evaluate on test data
    predictions = lr_model.transform(test_df)
    predictions.select("label", "prediction", "probability").show(20, truncate=False)

    # Binary metrics: AUC (ROC)
    binary_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc = binary_eval.evaluate(predictions)

    # Accuracy using MulticlassClassificationEvaluator
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = acc_eval.evaluate(predictions)

    print("=== Evaluation Metrics ===")
    print("Accuracy:", accuracy)
    print("Area Under ROC (AUC):", auc)

    # 8. (Later) These metrics will be written to HBase
    # For now we just print them so they can be captured in screenshots
    # Example values to use later when writing to HBase:
    # row_key = "titanic_model_1"
    # cf:accuracy = str(accuracy)
    # cf:auc = str(auc)

    spark.stop()


if __name__ == "__main__":
    main()
