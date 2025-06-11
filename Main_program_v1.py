# Import all necessary libraries for this program
from pathlib import Path
import kagglehub
import pandas as pd
import os
import pickle
from pathlib import Path
import joblib

# Download dataset
path = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")

# Read CSV files
file_path1 = os.path.join(path, "1429_1.csv")
file_path2 = os.path.join(path, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
file_path3 = os.path.join(path, "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)

# Load pickle mapping

pickle_file_path = Path.cwd() / "Clustering_model" / "unique_categories_dict.pkl"
if not pickle_file_path.exists():
    raise FileNotFoundError(f"Missing pickle file: {pickle_file_path}")
with open(pickle_file_path, "rb") as f:
    meta_category_mapping = pickle.load(f)

# Filter and prepare data
columns_df1 = ['name', 'asins', 'categories', 'reviews.doRecommend', 'reviews.numHelpful', 'reviews.rating', 'reviews.text', 'reviews.title']
columns_other = columns_df1 + ['imageURLs']
df1_filtered = df1[columns_df1]
df1_filtered['imageURLs'] = "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"
df2_filtered = df2[columns_other]
df3_filtered = df3[columns_other]

# Combine the dataframes, and map the meta-categories that we extracted via the clustering model to the 'categories' column
df_combined = pd.concat([df1_filtered, df2_filtered, df3_filtered], ignore_index=True)
df_combined['meta_category'] = df_combined['categories'].map(meta_category_mapping).fillna("Unknown")

# Print output of the combined dataframe
print("Shape of the combined dataframe:", df_combined.shape)
print(df_combined.head())

# This is pseudo code, to be completed later
# First we need to clean the data
# Then we convert the relevant columns of the dataframe to word embeddings
# Afterwards, we import the classifier model, the clustering-classification model and the product scoring/reviewing model
# We then put all the models together in a single function that takes as input a user query (in the shape of one of the six meta-categories) 
# and returns the top 3 products for that meta-category, including the product name, image URL, aggregated rating, and a short summary of the reviews
# finally, we deploy the function as a web service using Gradio
