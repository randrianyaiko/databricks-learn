from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from mlflow.models.signature import infer_signature
import mlflow
import os

# -----------------------------
# Load features
# -----------------------------
spark_df = spark.read.table("main.gold.features_transactions")

# -----------------------------
# Categorical encoding
# -----------------------------
categorical_cols = ["country", "bin_country", "channel", "merchant_category"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical_cols]

# -----------------------------
# Feature assembly
# -----------------------------
feature_cols = [
    "account_age_days",
    "total_transactions_user",
    "avg_amount_user",
    "amount_log",
    "distance_log",
    "amount_vs_user_avg",
    "shipping_distance_km"
] + [f"{c}_ohe" for c in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# -----------------------------
# Handle class imbalance
# -----------------------------
fraud_ratio = spark_df.filter(col("is_fraud") == 1).count() / spark_df.count()
spark_df = spark_df.withColumn(
    "class_weight",
    col("is_fraud") * (1.0 / fraud_ratio) + (1 - col("is_fraud")) * (1.0 / (1 - fraud_ratio))
)

# -----------------------------
# Define model and pipeline
# -----------------------------
gbt = GBTClassifier(featuresCol="features", labelCol="is_fraud", weightCol="class_weight", maxIter=50)
pipeline = Pipeline(stages=indexers + encoders + [assembler, gbt])

# -----------------------------
# Train / test split
# -----------------------------
train_df, test_df = spark_df.randomSplit([0.8, 0.2], seed=42)

# -----------------------------
# Set up MLflow experiment (create if not exists)
# -----------------------------
experiment_name = "/Users/rindranyaiko@gmail.com/handson-databricks-fraud_detection"
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

# -----------------------------
# MLflow tracking directory
# -----------------------------
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/main/gold/mlflow_volume/tmp/mlflow"
dbutils.fs.mkdirs("/Volumes/main/gold/mlflow_volume/tmp/mlflow")

# -----------------------------
# Train, evaluate, log model
# -----------------------------
with mlflow.start_run(run_name="gbt_fraud_model") as run:

    # Fit pipeline
    pipeline_model = pipeline.fit(train_df)

    # Predict
    predictions = pipeline_model.transform(test_df)

    # Metrics
    roc_auc = BinaryClassificationEvaluator(
        labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    pr_auc = BinaryClassificationEvaluator(
        labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(predictions)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    # Signature
    sample_input = test_df.limit(5).toPandas()[[
        "account_age_days",
        "total_transactions_user",
        "avg_amount_user",
        "amount_log",
        "distance_log",
        "amount_vs_user_avg",
        "shipping_distance_km",
    ] + categorical_cols]
    sample_output = pipeline_model.transform(test_df.limit(5)).select("prediction").toPandas()
    signature = infer_signature(sample_input, sample_output)

    # Log model
    mlflow.spark.log_model(
        pipeline_model,
        artifact_path="model",
        signature=signature,
        input_example=sample_input,
        dfs_tmpdir="/Volumes/main/gold/mlflow_volume/tmp/mlflow"
    )

    # Save run_id
    run_id = run.info.run_id
    dbutils.fs.put("/Volumes/main/gold/mlflow_volume/latest_run_id.txt", run_id, overwrite=True)

print(f"Training complete. Run ID: {run_id}, ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
