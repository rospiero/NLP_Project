# Import all necessary libraries for this program
from pathlib import Path
import kagglehub
import pandas as pd
import os
import pickle
from pathlib import Path
import joblib
import re
from sentence_transformers import SentenceTransformer

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


# First we need to clean the data with our data cleaning function, taking into account that the reviews.text column is a string but might contain floats
# Clean the review text column and append the cleaned text to a new column, given the size of the dataset this will not increase the model runtime significantly
def light_clean(text):
    if isinstance(text, float):
        text = str(text)  # Convert float to string
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text

df_combined["cleaned_text"] = df_combined["reviews.text"].apply(light_clean)

# Then we convert the relevant columns of the dataframe to word embeddings using a pre-trained sentence transformer model. We use a sentence transformer as these are designed 
# to work well with natural, unaltered sentences and are trained on a large corpus of text data, making them suitable for generating embeddings for a wide range of text inputs.

model = SentenceTransformer('all-MiniLM-L6-v2')
# Generate embeddings for the cleaned review text
embeddings = model.encode(df_combined["cleaned_text"].tolist(), show_progress_bar=True, convert_to_tensor= True, device='cuda')
# Append the embeddings to the dataframe
df_combined["embeddings"] = embeddings.tolist()

# Import the random forest classifier model
rf_classifier_model_path = Path.cwd() / "Joblib_files" / "classifier_random_forest_model.joblib"
if not rf_classifier_model_path.exists():
    raise FileNotFoundError(f"Missing classifier model file: {rf_classifier_model_path}")
rf_classifier_model = joblib.load(rf_classifier_model_path)
# Import the clustering-classification model
clustering_model_path = Path.cwd() / "Joblib_files" / "Clustering_logistic_regression_model_cv.joblib"
if not clustering_model_path.exists():
    raise FileNotFoundError(f"Missing clustering model file: {clustering_model_path}")
clustering_model = joblib.load(clustering_model_path)

# Define a function below to get the top 3 products for a given meta-category as input. The function uses the rf_classifier_model to create sentiment analysis 
# on the embeddings of the reviews



# We then put all the models together in a single function that takes as input a user query (in the shape of one of the six meta-categories) 
# and returns the top 3 products for that meta-category, including the product name, image URL, aggregated rating, and a short summary of the reviews
# finally, we deploy the function as a web service using Gradio
