import datetime
import sqlite3

import pandas as pd


def store_csv(collection_features, csv_path: str) -> None:
    """
    Stores the audio feature collection into a CSV file.

    :param collection_features: List of feature dictionaries
    :param csv_path: File path for the CSV
    :param date_prefix: Whether to add a date prefix to the file name
    """
    df = pd.DataFrame(collection_features)
    df.to_csv(csv_path, index=False)
    print(f"Data successfully stored in CSV at: {csv_path}")


def store_parquet(collection_features, parquet_path: str) -> None:
    """
    Stores the audio feature collection into a Parquet file.

    :param collection_features: List of feature dictionaries
    :param parquet_path: File path for the Parquet file
    """
    df = pd.DataFrame(collection_features)
    df.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"Data successfully stored in Parquet at: {parquet_path}")


def store_sqlite(
    collection_features, db_path: str, table_name: str = "audio_features"
) -> None:
    """
    Stores the audio feature collection into an SQLite database.

    :param collection_features: List of feature dictionaries
    :param db_path: File path for the SQLite database (e.g. 'features.db')
    :param table_name: Name of the table to store the data
    """
    df = pd.DataFrame(collection_features)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Data successfully stored in SQLite at: {db_path}, table: {table_name}")
