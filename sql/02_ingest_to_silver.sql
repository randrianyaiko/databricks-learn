-- Ingestion: Load raw CSV into Silver Delta table

CREATE OR REPLACE TABLE main.silver.transaction
AS
SELECT *
FROM csv.`/Workspace/Users/rindranyaiko@gmail.com/handson-databricks-fraud_detection/data/transactions.csv`;
