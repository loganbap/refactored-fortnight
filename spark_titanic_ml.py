import happybase
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

    # 3. More basic cleaning: drop rows with nulls in label or features
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

    # 8. Write metrics to HBase using happybase
    try:
        connection = happybase.Connection('master')   # Thrift server host
        connection.open()
        table = connection.table('titanic_metrics')

        row_key = b"titanic_model_1"
        data = {
            b"cf:accuracy": str(accuracy).encode("utf-8"),
            b"cf:auc":      str(auc).encode("utf-8")
        }

        table.put(row_key, data)
        print("Successfully wrote metrics to HBase table 'titanic_metrics' for row 'titanic_model_1'.")
    except Exception as e:
        print("Error writing metrics to HBase:", e)
    finally:
        try:
            connection.close()
        except Exception:
            pass

    spark.stop()



if __name__ == "__main__":
    main()