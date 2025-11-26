-- Create catalogs and schemas if they do not exist

CREATE CATALOG IF NOT EXISTS main;

CREATE SCHEMA IF NOT EXISTS main.bronze;
CREATE SCHEMA IF NOT EXISTS main.silver;
CREATE SCHEMA IF NOT EXISTS main.gold;

-- Optional volumes for MLflow
CREATE VOLUME IF NOT EXISTS main.gold.mlflow_volume;
