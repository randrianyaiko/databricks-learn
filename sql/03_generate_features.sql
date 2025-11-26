CREATE OR REPLACE TABLE main.gold.features_transactions AS
SELECT
    *,
    -- Log transforms
    LOG1P(amount) AS amount_log,
    LOG1P(shipping_distance_km) AS distance_log,

    -- Feature ratios
    amount / (avg_amount_user + 1e-6) AS amount_vs_user_avg,
    total_transactions_user / (account_age_days + 1) AS user_frequency,

    -- Time features
    HOUR(transaction_time) AS hour,
    DAYOFWEEK(transaction_time) AS day_of_week,
    CASE WHEN DAYOFWEEK(transaction_time) > 5 THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN HOUR(transaction_time) <= 5 THEN 1 ELSE 0 END AS is_night,

    -- Categorical mismatch
    CASE WHEN country != bin_country THEN 1 ELSE 0 END AS country_mismatch

FROM main.silver.transaction;
