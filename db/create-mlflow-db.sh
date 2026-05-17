#!/bin/bash
# Create MLflow database alongside the main sentinel database
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $POSTGRES_USER;
EOSQL

echo "MLflow database created successfully"
