"""
Script to export BigQuery data to CSV files for local testing.
Run this on GCP to export the data you need locally.
"""
from google.cloud import bigquery
import pandas as pd

# Initialize BigQuery client with YOUR project (students-group2)
# Jobs will run in your project, but query data from master-ai-cloud
client = bigquery.Client(project='students-group2')

# Export movies
print("Exporting movies...")
movies_query = """
SELECT movieId, title, genres
FROM `master-ai-cloud.MoviePlatform.movies`
"""
movies_df = client.query(movies_query).to_dataframe()
movies_df.to_csv('movies.csv', index=False)
print(f"Exported {len(movies_df)} movies to movies.csv")

# Export ratings (limit to 100k for reasonable file size)
print("Exporting ratings...")
ratings_query = """
SELECT userId, movieId, rating, timestamp
FROM `master-ai-cloud.MoviePlatform.ratings`
LIMIT 100000
"""
ratings_df = client.query(ratings_query).to_dataframe()
ratings_df.to_csv('ratings.csv', index=False)
print(f"Exported {len(ratings_df)} ratings to ratings.csv")

print("\nDone! Download these files to your local machine.")
print("\nTo download from JupyterLab:")
print("1. Right-click on movies.csv → Download")
print("2. Right-click on ratings.csv → Download")