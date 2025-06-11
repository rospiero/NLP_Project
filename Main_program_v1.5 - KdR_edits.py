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
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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

# Define a Class below to score the top 3 products for a given meta-category. The Class contains functions that use the rf_classifier_model to create sentiment analysis 
# on the embeddings of the reviews
class ProductScoringSystem:
    def __init__(self, df_combined, model_filename = 'Classifier_random_forest_model.joblib'):
        self.df = df_combined.copy()
        self.product_scores = None
        self.model_filename = model_filename
        self.rf_model = None
        self._load_model()

    def _load_model(self):
        try:
            model_path = Path.cwd() / 'Joblib_files' / self.model_filename
            self.rf_model = joblib.load(model_path)
            print(f"Successfully loaded RandomForest model from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find model file at {Path.cwd() / 'Joblib_files' / self.model_filename}")
            print("Sentiment prediction will be skipped.")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Sentiment prediction will be skipped.")

    def calculate_review_quality_scores(self):
        """
        Calculate quality scores for individual reviews based on:
        - Text length (longer reviews often more informative)
        - Helpfulness votes
        - Semantic coherence (using embeddings)
        """
        print("Calculating review quality scores...")

        # Drop rows with missing review text or rating *before* calculating quality scores
        self.df = self.df.dropna(subset=["reviews.text", "reviews.rating"])
        print(f"Dropped rows with missing reviews.text or reviews.rating. Remaining rows: {len(self.df)}")


        # Text length score (normalized)
        self.df['text_length'] = self.df['cleaned_text'].str.len()
        self.df['length_score'] = (
            (self.df['text_length'] - self.df['text_length'].min()) /
            (self.df['text_length'].max() - self.df['text_length'].min())
        ).clip(0, 1)

        # Helpfulness score (normalized)
        self.df['reviews.numHelpful'] = self.df['reviews.numHelpful'].fillna(0)
        max_helpful = self.df['reviews.numHelpful'].max()
        if max_helpful > 0:
            self.df['helpfulness_score'] = (
                self.df['reviews.numHelpful'] / max_helpful
            ).clip(0, 1)
        else:
            self.df['helpfulness_score'] = 0.0

        # Semantic coherence score
        # Reviews with embeddings very different from product average might be outliers/spam
        self.df['coherence_score'] = 0.5  # Default neutral score

        for asin in self.df['asins'].unique():
            product_reviews = self.df[self.df['asins'] == asin].copy() # Use .copy()
            if len(product_reviews) > 1 and 'embeddings' in product_reviews.columns:
                try:
                    # Get embeddings for this product's reviews
                    embeddings = np.array([
                        emb for emb in product_reviews['embeddings']
                        if emb is not None and len(emb) > 0
                    ])

                    if len(embeddings) > 1:
                        # Calculate average embedding for the product
                        avg_embedding = np.mean(embeddings, axis=0)

                        # Calculate similarity of each review to product average
                        similarities = cosine_similarity(embeddings, [avg_embedding]).flatten()

                        # Update coherence scores - Need to map back to original df indices
                        valid_indices = product_reviews.index[
                            product_reviews['embeddings'].apply(
                                lambda x: x is not None and len(x) > 0
                            )
                        ]
                        self.df.loc[valid_indices, 'coherence_score'] = similarities
                except Exception as e:
                    print(f"Warning: Could not calculate coherence for product {asin}: {e}")

        # Combined quality score (weighted average)
        self.df['review_quality_score'] = (
            0.3 * self.df['length_score'] +
            0.4 * self.df['helpfulness_score'] +
            0.3 * self.df['coherence_score']
        )

        return self.df

    def predict_sentiment_scores(self):
        """
        Use the trained RandomForest model to predict sentiment scores from embeddings
        """
        print("Predicting sentiment scores using RandomForest model...")

        # Initialize sentiment score columns
        # Ensure these columns exist before assigning values
        if 'rf_sentiment_prediction' not in self.df.columns:
            self.df['rf_sentiment_prediction'] = 0.0
        if 'rf_sentiment_confidence' not in self.df.columns:
            self.df['rf_sentiment_confidence'] = 0.0
        if 'rf_sentiment_probabilities' not in self.df.columns:
             self.df['rf_sentiment_probabilities'] = None


        if self.rf_model is None:
            print("No model available - using default neutral sentiment scores")
            self.df['rf_sentiment_prediction'] = 0.0  # Neutral
            self.df['rf_sentiment_confidence'] = 0.0
            return self.df

        # Get valid embeddings
        valid_embedding_mask = self.df['embeddings'].apply(
            lambda x: x is not None and len(x) > 0
        )

        if not valid_embedding_mask.any():
            print("Warning: No valid embeddings found for sentiment prediction")
            # Still need to ensure sentiment columns exist even if no predictions are made
            if 'rf_sentiment_normalized' not in self.df.columns:
                 self.df['rf_sentiment_normalized'] = 0.0
            return self.df

        valid_indices = self.df[valid_embedding_mask].index
        valid_embeddings = np.array([
            self.df.loc[idx, 'embeddings'] for idx in valid_indices
        ])

        try:
            # Make predictions
            predictions = self.rf_model.predict(valid_embeddings)

            # Get prediction probabilities if available
            if hasattr(self.rf_model, 'predict_proba'):
                probabilities = self.rf_model.predict_proba(valid_embeddings)

                # Calculate confidence as max probability
                confidences = np.max(probabilities, axis=1)

                # Store probabilities for later analysis
                # Ensure the target column can hold lists/arrays
                self.df.loc[valid_indices, 'rf_sentiment_probabilities'] = list(probabilities)
                self.df.loc[valid_indices, 'rf_sentiment_confidence'] = confidences
            else:
                # If no probabilities available, use binary confidence
                self.df.loc[valid_indices, 'rf_sentiment_confidence'] = 1.0

            # Store predictions
            self.df.loc[valid_indices, 'rf_sentiment_prediction'] = predictions

            # Convert predictions to normalized sentiment score
            self.df['rf_sentiment_normalized'] = self._normalize_sentiment_predictions(
                self.df.loc[valid_indices, 'rf_sentiment_prediction'] # Apply normalization only to predicted values
            )
            # Fill NaNs in the new normalized column for rows without valid embeddings
            self.df['rf_sentiment_normalized'] = self.df['rf_sentiment_normalized'].fillna(0.0)


            print(f"Successfully predicted sentiment for {len(valid_indices)} reviews")

        except Exception as e:
            print(f"Error during sentiment prediction: {e}")
            # Fallback to neutral scores in case of prediction error
            self.df['rf_sentiment_prediction'] = 0.0
            self.df['rf_sentiment_confidence'] = 0.0
            self.df['rf_sentiment_normalized'] = 0.0

        return self.df

    def _normalize_sentiment_predictions(self, predictions):
        """
        Normalize sentiment predictions to [-1, 1] scale where:
        -1 = very negative, 0 = neutral, 1 = very positive

        Adjust this method based on your model's output format
        """
        # Ensure predictions is a pandas Series to use .map() or vectorization
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions)

        # If your model outputs class labels (0, 1, 2):
        if predictions.dtype in ['int64', 'int32'] or all(pred in [0, 1, 2] for pred in predictions.dropna()):
            # Convert class labels to sentiment scores
            mapping = {0: -1.0, 1: 0.0, 2: 1.0}  # negative, neutral, positive
            return predictions.map(mapping).fillna(0.0)

        # If your model already outputs continuous sentiment scores:
        # Check bounds only on non-NaN values
        preds_no_na = predictions.dropna()
        if not preds_no_na.empty and preds_no_na.min() >= 0 and preds_no_na.max() <= 1:
            # Convert [0,1] to [-1,1]
            return (predictions * 2) - 1

        elif not preds_no_na.empty and preds_no_na.min() >= -1 and preds_no_na.max() <= 1:
            # Already in [-1,1] range
            return predictions.fillna(0.0) # Fill potential NaNs

        else:
            # Generic normalization to [-1,1]
            min_pred, max_pred = predictions.min(), predictions.max()
            if max_pred != min_pred:
                # Apply normalization only to non-NaN values
                normalized = predictions.apply(lambda x: 2 * (x - min_pred) / (max_pred - min_pred) - 1 if pd.notna(x) else np.nan)
                return normalized.fillna(0.0)
            else:
                # If all predictions are the same, they are likely neutral (or just one value)
                return pd.Series([0.0] * len(predictions), index=predictions.index).fillna(0.0)


    def aggregate_product_scores(self):
        """
        Aggregate review-level data to product-level scores
        """
        print("Aggregating product scores...")

        product_aggregations = []

        # Iterate over unique asins *after* dropping rows
        # Ensure the 'asins' column exists in the current self.df
        if 'asins' not in self.df.columns:
            print("Error: 'asins' column not found in the DataFrame.")
            self.product_scores = pd.DataFrame() # Return empty dataframe
            return self.product_scores

        # Get unique asins from the current state of the DataFrame
        unique_asins_after_cleaning = self.df['asins'].unique()

        for asin in unique_asins_after_cleaning:
            # Filter for reviews corresponding to the current ASIN
            product_data = self.df[self.df['asins'] == asin].copy() # Use .copy()

            # Check if the product_data DataFrame is not empty
            # This handles cases where an ASIN existed before cleaning but has no reviews left
            if not product_data.empty:
                # Basic product info - use .iloc[0] only on non-empty dataframe
                product_name = product_data['name'].iloc[0]
                meta_category = product_data['meta_category'].iloc[0]
                review_count = len(product_data)

                # Rating aggregations
                # Ensure 'reviews.rating' exists and handle potential NaNs before mean/median/std
                if 'reviews.rating' in product_data.columns:
                    avg_rating = product_data['reviews.rating'].mean()
                    median_rating = product_data['reviews.rating'].median()
                    rating_std = product_data['reviews.rating'].std()
                else:
                    avg_rating, median_rating, rating_std = np.nan, np.nan, np.nan


                # Quality-weighted rating (weight by review quality)
                # Ensure 'review_quality_score' exists and handle potential NaNs/zeros in weights
                if 'review_quality_score' in product_data.columns and not product_data['review_quality_score'].isnull().all():
                    quality_weights = product_data['review_quality_score']
                    # Handle cases where all weights are zero or NaN - use simple average instead
                    if quality_weights.sum() > 0:
                         quality_weighted_rating = np.average(
                            product_data['reviews.rating'].dropna(), # Only average valid ratings
                            weights=quality_weights[product_data['reviews.rating'].notna()] # Match weights to non-NaN ratings
                         )
                    else:
                         quality_weighted_rating = product_data['reviews.rating'].mean() # Fallback to simple mean
                else:
                    quality_weighted_rating = product_data['reviews.rating'].mean() # Fallback to simple mean

                # RandomForest sentiment aggregations
                # Ensure sentiment columns exist
                avg_rf_sentiment = product_data['rf_sentiment_normalized'].mean() if 'rf_sentiment_normalized' in product_data.columns else 0.0
                avg_rf_confidence = product_data['rf_sentiment_confidence'].mean() if 'rf_sentiment_confidence' in product_data.columns else 0.0


                # Quality-weighted sentiment score (weight by both quality and confidence)
                # Ensure required columns exist and handle potential NaNs/zeros in weights
                if ('review_quality_score' in product_data.columns and
                    'rf_sentiment_confidence' in product_data.columns and
                    'rf_sentiment_normalized' in product_data.columns):

                    combined_weights = product_data['review_quality_score'] * product_data['rf_sentiment_confidence']
                    # Only consider weights and sentiments where sentiment is not NaN
                    valid_sentiment_mask = product_data['rf_sentiment_normalized'].notna()
                    combined_weights = combined_weights[valid_sentiment_mask]
                    valid_sentiments = product_data['rf_sentiment_normalized'][valid_sentiment_mask]


                    if combined_weights.sum() > 0 and len(valid_sentiments) > 0:
                        quality_weighted_sentiment = np.average(
                            valid_sentiments,
                            weights=combined_weights
                        )
                    else:
                        # Fallback if weights sum to 0 or no valid sentiments
                        quality_weighted_sentiment = avg_rf_sentiment
                else:
                     quality_weighted_sentiment = avg_rf_sentiment


                # Sentiment distribution
                # Ensure 'rf_sentiment_normalized' exists before calculating ratios
                if 'rf_sentiment_normalized' in product_data.columns and review_count > 0:
                    positive_reviews = (product_data['rf_sentiment_normalized'] > 0.1).sum()
                    negative_reviews = (product_data['rf_sentiment_normalized'] < -0.1).sum()
                    neutral_reviews = review_count - positive_reviews - negative_reviews

                    sentiment_distribution = {
                        'positive_ratio': positive_reviews / review_count,
                        'negative_ratio': negative_reviews / review_count,
                        'neutral_ratio': neutral_reviews / review_count
                    }
                else:
                    # Default to neutral distribution if no sentiment data or reviews
                    sentiment_distribution = {
                        'positive_ratio': 0.0,
                        'negative_ratio': 0.0,
                        'neutral_ratio': 1.0 if review_count > 0 else 0.0
                    }


                # Recommendation ratio
                recommend_ratio = product_data['reviews.doRecommend'].mean() if 'reviews.doRecommend' in product_data.columns else 0.5

                # Review volume penalty/boost (handle class imbalance)
                volume_factor = self._calculate_volume_factor(review_count)

                # Calculate final composite score
                composite_score = self._calculate_composite_score(
                    quality_weighted_rating if pd.notna(quality_weighted_rating) else (avg_rating if pd.notna(avg_rating) else 3.0), # Fallback rating
                    quality_weighted_sentiment,
                    recommend_ratio,
                    volume_factor,
                    rating_std if pd.notna(rating_std) else 0.5, # Fallback std
                    avg_rf_confidence
                )

                product_aggregations.append({
                    'asin': asin,
                    'product_name': product_name,
                    'meta_category': meta_category,
                    'review_count': review_count,
                    'avg_rating': avg_rating,
                    'median_rating': median_rating,
                    'rating_std': rating_std,
                    'quality_weighted_rating': quality_weighted_rating,
                    'avg_rf_sentiment': avg_rf_sentiment,
                    'quality_weighted_sentiment': quality_weighted_sentiment,
                    'avg_rf_confidence': avg_rf_confidence,
                    'positive_ratio': sentiment_distribution['positive_ratio'],
                    'negative_ratio': sentiment_distribution['negative_ratio'],
                    'neutral_ratio': sentiment_distribution['neutral_ratio'],
                    'recommend_ratio': recommend_ratio,
                    'volume_factor': volume_factor,
                    'composite_score': composite_score
                })
            else:
                print(f"Warning: No reviews remaining for ASIN {asin} after cleaning. Skipping aggregation.")


        self.product_scores = pd.DataFrame(product_aggregations)
        return self.product_scores


    def _calculate_volume_factor(self, review_count):
        """
        Calculate volume adjustment factor to handle class imbalance
        """
        if review_count < 5:
            return 0.7  # Significant penalty for very few reviews
        elif review_count < 10:
            return 0.85  # Moderate penalty
        elif review_count < 50:
            return 1.0   # No adjustment
        elif review_count < 100:
            return 1.05  # Slight boost
        else:
            return 1.1   # Small boost for high-volume products

    def _calculate_composite_score(self, rating, sentiment_score, recommend_ratio,
                                 volume_factor, rating_std, avg_confidence):
        """
        Calculate final composite score combining all factors
        """
        # Normalize rating to 0-1 scale (assuming 1-5 rating scale)
        # Ensure rating is a number before calculation, fallback to 3 if NaN
        normalized_rating = ((rating if pd.notna(rating) else 3.0) - 1) / 4

        # Normalize sentiment score from [-1,1] to [0,1]
        # Ensure sentiment_score is a number, fallback to 0 if NaN
        normalized_sentiment = ((sentiment_score if pd.notna(sentiment_score) else 0.0) + 1) / 2

        # Consistency bonus (lower std deviation is better)
        # Ensure rating_std is a number, fallback to 0.5 if NaN or 0
        consistency_factor = 1 - min((rating_std if pd.notna(rating_std) and rating_std > 0 else 0.5) / 4, 0.3)

        # Confidence bonus (higher confidence in predictions is better)
        # Ensure avg_confidence is a number, fallback to 0.8 if NaN
        confidence_factor = 0.8 + (0.2 * (avg_confidence if pd.notna(avg_confidence) else 0.8)) # Scale from 0.8 to 1.0

        # Weighted combination
        composite = (
            0.35 * normalized_rating +          # Base rating weight (slightly reduced)
            0.35 * normalized_sentiment +       # Sentiment score weight (increased)
            0.2 * recommend_ratio +             # Recommendation weight
            0.1 * consistency_factor            # Consistency bonus
        )

        # Apply volume factor and confidence factor
        final_score = composite * volume_factor * confidence_factor

        return final_score


    def get_top_products_by_category(self, meta_category, top_k=3):
        """
        Get top K products for a specific meta-category
        """
        if self.product_scores is None:
            raise ValueError("Must run calculate_scores() first")

        # Ensure meta_category column exists before filtering
        if 'meta_category' not in self.product_scores.columns:
             print("Error: 'meta_category' column not found in product scores.")
             return pd.DataFrame()

        # Filter and ensure the category exists in the scores
        if meta_category not in self.product_scores['meta_category'].unique():
             print(f"Warning: Meta-category '{meta_category}' not found in product scores.")
             return pd.DataFrame()


        category_products = self.product_scores[
            self.product_scores['meta_category'] == meta_category
        ].copy()

        # Sort by composite score
        category_products = category_products.sort_values(
            'composite_score', ascending=False
        )

        return category_products.head(top_k)

    def calculate_scores(self):
        """
        Run the complete scoring pipeline
        """
        print("Starting product scoring pipeline...")

        # Step 1: Calculate review quality scores
        self.calculate_review_quality_scores()

        # Step 2: Predict sentiment using RandomForest model
        self.predict_sentiment_scores()

        # Step 3: Aggregate to product level
        self.aggregate_product_scores()

        print(f"Scoring complete! Processed {len(self.product_scores)} products.")
        return self.product_scores

    def get_scoring_summary(self):
        """
        Get summary statistics of the scoring results
        """
        if self.product_scores is None or self.product_scores.empty:
            return "No scores calculated yet or dataframe is empty. Run calculate_scores() first."

        summary = {
            'total_products': len(self.product_scores),
            'categories': self.product_scores['meta_category'].value_counts().to_dict(),
            'avg_composite_score': self.product_scores['composite_score'].mean(),
            'avg_sentiment_score': self.product_scores['avg_rf_sentiment'].mean(),
            'avg_confidence': self.product_scores['avg_rf_confidence'].mean(),
            'score_distribution': self.product_scores['composite_score'].describe().to_dict(),
            'sentiment_distribution': {
                'avg_positive_ratio': self.product_scores['positive_ratio'].mean(),
                'avg_negative_ratio': self.product_scores['negative_ratio'].mean(),
                'avg_neutral_ratio': self.product_scores['neutral_ratio'].mean()
            }
        }

        return summary

    def get_sentiment_analysis_summary(self):
        """
        Get detailed sentiment analysis summary
        """
        if self.df is None or self.df.empty:
            return "No data available."

        total_reviews = len(self.df)
        valid_predictions = self.df['rf_sentiment_prediction'].notna().sum() if 'rf_sentiment_prediction' in self.df.columns else 0

        # Ensure sentiment columns exist for distribution calculation
        positive_count = (self.df['rf_sentiment_normalized'] > 0.1).sum() if 'rf_sentiment_normalized' in self.df.columns else 0
        negative_count = (self.df['rf_sentiment_normalized'] < -0.1).sum() if 'rf_sentiment_normalized' in self.df.columns else 0
        # Calculate neutral based on total reviews if possible, otherwise use what's available
        neutral_count = total_reviews - positive_count - negative_count if 'rf_sentiment_normalized' in self.df.columns else 0


        sentiment_summary = {
            'total_reviews': total_reviews,
            'valid_predictions': valid_predictions,
            'coverage': valid_predictions / total_reviews if total_reviews > 0 else 0,
            'avg_confidence': self.df['rf_sentiment_confidence'].mean() if 'rf_sentiment_confidence' in self.df.columns else np.nan,
            'sentiment_distribution': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        }

        return sentiment_summary

# Test the functions above
# Basic usage
scorer = ProductScoringSystem(df_combined)
product_scores = scorer.calculate_scores()

top_portable_electronics = scorer.get_top_products_by_category('Portable Electronics', top_k=3)
print(top_portable_electronics)

# Get detailed summaries
scoring_summary = scorer.get_scoring_summary()
sentiment_summary = scorer.get_sentiment_analysis_summary()

# We then put all the models together in a single function that takes as input a user query (in the shape of one of the six meta-categories) 
# and returns the top 3 products for that meta-category, including the product name, image URL, aggregated rating, and a short summary of the reviews
# finally, we deploy the function as a web service using Gradio
