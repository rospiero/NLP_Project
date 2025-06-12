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
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

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
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleProductScorer:
    def __init__(self, df, model_path='random_forest_model.joblib'):
        self.df = df.copy()
        self.product_scores = None
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path):
        """Load the trained model if available"""
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except:
            print("No model found - using basic sentiment scoring")
            return None
    
    def _get_sentiment_score(self, row):
        """Get sentiment score from model or fallback to rating-based score"""
        if self.model and 'embeddings' in row and row['embeddings'] is not None:
            try:
                # Use model prediction if available
                prediction = self.model.predict([row['embeddings']])[0]
                # Convert to -1 to 1 scale (adjust based on your model output)
                if prediction in [0, 1, 2]:  # Class labels
                    return {0: -1, 1: 0, 2: 1}[prediction]
                else:
                    return max(-1, min(1, prediction))  # Clamp to [-1, 1]
            except:
                pass
        
        # Fallback: convert rating to sentiment (-1 to 1)
        rating = row.get('reviews.rating', 3)
        return (rating - 3) / 2  # 1->-1, 3->0, 5->1
    
    def calculate_scores(self):
        """Calculate product scores using simple, interpretable metrics"""
        print("Calculating product scores...")
        
        # Add sentiment scores to reviews
        self.df['sentiment_score'] = self.df.apply(self._get_sentiment_score, axis=1)
        
        # Group by product and calculate aggregate scores
        product_scores = []
        
        for asin in self.df['asins'].dropna().unique():
            product_data = self.df[self.df['asins'] == asin]
    
            if product_data.empty:
                continue  # Skip empty group

            product_name = product_data['name'].iloc[0]
            if pd.isna(product_name):
                print(f"Skipping ASIN {asin} due to missing product name.")
                continue  # Skip products with no name

            # Basic metrics
            review_count = len(product_data)
            avg_rating = product_data['reviews.rating'].mean()
            avg_sentiment = product_data['sentiment_score'].mean()
            
            # Helpfulness score (higher is better)
            helpfulness = product_data['reviews.numHelpful'].fillna(0).mean()
            
            # Recommendation ratio
            recommend_ratio = product_data['reviews.doRecommend'].mean() if 'reviews.doRecommend' in product_data.columns else 0.5
            
            # Simple composite score (0-1 scale)
            # Weight: 40% rating, 30% sentiment, 20% recommendations, 10% helpfulness
            rating_score = (avg_rating - 1) / 4  # Convert 1-5 to 0-1
            sentiment_score = (avg_sentiment + 1) / 2  # Convert -1,1 to 0-1
            helpfulness_score = min(helpfulness / 10, 1.0)  # Cap at 1.0
            
            composite_score = (
                0.4 * rating_score + 
                0.3 * sentiment_score + 
                0.2 * recommend_ratio + 
                0.1 * helpfulness_score
            )
            
            # Apply review count penalty for products with very few reviews
            if review_count < 5:
                composite_score *= 0.8
            elif review_count < 10:
                composite_score *= 0.9
            
            product_scores.append({
                'asin': asin,
                'product_name': product_data['name'].iloc[0],
                'category': product_data['meta_category'].iloc[0],
                'review_count': review_count,
                'avg_rating': avg_rating,
                'avg_sentiment': avg_sentiment,
                'recommend_ratio': recommend_ratio,
                'helpfulness': helpfulness,
                'composite_score': composite_score
            })
        
        self.product_scores = pd.DataFrame(product_scores)
        self.product_scores = self.product_scores.sort_values('composite_score', ascending=False)
        
        print(f"Scored {len(self.product_scores)} products")
        return self.product_scores
    
    def get_top_products(self, category=None, top_k=3):
        """Get top K products overall or by category"""
        if self.product_scores is None:
            raise ValueError("Run calculate_scores() first")
        
        if category:
            filtered_scores = self.product_scores[self.product_scores['category'] == category]
        else:
            filtered_scores = self.product_scores
        
        return filtered_scores.head(top_k)
    
    def get_top_products_by_category(self, category, top_k=3):
        """Alias for get_top_products for backward compatibility"""
        return self.get_top_products(category, top_k)
    
    def get_summary(self):
        """Get simple summary of results"""
        if self.product_scores is None:
            return "No scores calculated yet"
        
        return {
            'total_products': len(self.product_scores),
            'avg_score': self.product_scores['composite_score'].mean(),
            'avg_rating': self.product_scores['avg_rating'].mean(),
            'avg_sentiment': self.product_scores['avg_sentiment'].mean(),
            'categories': self.product_scores['category'].value_counts().to_dict()
        }
    
    def get_scoring_summary(self):
        """Get detailed summary (for backward compatibility)"""
        if self.product_scores is None:
            return "No scores calculated yet"
        
        return {
            'total_products': len(self.product_scores),
            'avg_composite_score': self.product_scores['composite_score'].mean(),
            'avg_sentiment_score': self.product_scores['avg_sentiment'].mean(),
            'categories': self.product_scores['category'].value_counts().to_dict(),
            'score_distribution': self.product_scores['composite_score'].describe().to_dict()
        }
    
    def get_sentiment_analysis_summary(self):
        """Get sentiment analysis summary (for backward compatibility)"""
        if self.df is None:
            return "No data available"
        
        total_reviews = len(self.df)
        valid_sentiments = self.df['sentiment_score'].notna().sum() if 'sentiment_score' in self.df.columns else 0
        
        return {
            'total_reviews': total_reviews,
            'valid_predictions': valid_sentiments,
            'coverage': valid_sentiments / total_reviews if total_reviews > 0 else 0,
            'sentiment_distribution': {
                'positive': (self.df['sentiment_score'] > 0.1).sum() if 'sentiment_score' in self.df.columns else 0,
                'negative': (self.df['sentiment_score'] < -0.1).sum() if 'sentiment_score' in self.df.columns else 0,
                'neutral': ((self.df['sentiment_score'] >= -0.1) & 
                           (self.df['sentiment_score'] <= 0.1)).sum() if 'sentiment_score' in self.df.columns else 0
            }
        }

    def print_top_products_by_category(self, top_k=3):
        """Print top products for each category"""
        print(f"\nTop {top_k} products by category:")
        print("=" * 50)
        
        for category in self.product_scores['category'].unique():
            top_products = self.get_top_products(category, top_k)
            print(f"\n{category.upper()}:")
            
            for i, (_, product) in enumerate(top_products.iterrows(), 1):
                print(f"{i}. {product['product_name'][:60]}")
                print(f"   Score: {product['composite_score']:.3f} | "
                      f"Rating: {product['avg_rating']:.1f} | "
                      f"Sentiment: {product['avg_sentiment']:.2f} | "
                      f"Reviews: {product['review_count']}")

# Usage:
# scorer, scores = score_products(df_combined)
# top_electronics = scorer.get_top_products('Electronics', top_k=3)

# Define a function that generates a summary of the reviews for the top 3 products in a given meta-category. The function takes as input the output of the get_top_products_by_category
#function from the ProductScoringSystem class, which is a dataframe with 3 rows and 17 columns. The function uses the OpenAI API to generate a summary of the reviews for each product,
#and returns a formatted string with the individual product summaries and a comparative conclusion.



# Test the functions above
# Basic usage
scorer = SimpleProductScorer(df_combined, model_path=rf_classifier_model_path)
product_scores = scorer.calculate_scores()

# Get detailed summaries
scoring_summary = scorer.get_scoring_summary()
sentiment_summary = scorer.get_sentiment_analysis_summary()

# Check if the category exists in the data first
available_categories = df_combined['meta_category'].unique()
print("Available categories:", available_categories)

# Use a category that actually exists in your data
target_category = 'Portable Electronics'  # Note: have user deliver input for category selection
if target_category not in available_categories:
    target_category = available_categories[0]  # Use first available category
    print(f"Using category: {target_category}")

try:
    top_products = scorer.get_top_products_by_category(target_category, top_k=3)
    print(f"\nTop 3 products in {target_category}:")
    print(top_products)
    
    # For testing, save the results to a CSV file
    output_path = Path.cwd() / f"Top_{target_category.replace(' ', '_')}.csv"
    top_products.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
except Exception as e:
    print(f"Error getting top products: {e}")
    # Fallback: show all available products
    print("\nAll available products:")
    print(scorer.product_scores.head(10))

# Get detailed summaries
try:
    scoring_summary = scorer.get_scoring_summary()
    sentiment_summary = scorer.get_sentiment_analysis_summary()
    
    print(f"\nScoring Summary:")
    print(f"Total products: {scoring_summary['total_products']}")
    print(f"Average composite score: {scoring_summary.get('avg_composite_score', 'N/A')}")
    print(f"Average sentiment score: {scoring_summary.get('avg_sentiment_score', 'N/A')}")
    
    print(f"\nSentiment Summary:")
    print(f"Total reviews: {sentiment_summary['total_reviews']}")
    print(f"Valid predictions: {sentiment_summary['valid_predictions']}")
    print(f"Coverage: {sentiment_summary['coverage']:.2%}")
    
except Exception as e:
    print(f"Error getting summaries: {e}")

# Debug: Print some basic info about the data
print(f"\nDebug Info:")
print(f"DataFrame shape: {df_combined.shape}")
print(f"Unique ASINs: {df_combined['asins'].nunique()}")
print(f"Meta categories: {df_combined['meta_category'].value_counts()}")
print(f"Sample product names: {df_combined['name'].head().tolist()}")

# Create a function that generates a summary of the reviews for the top 3 products in a given meta-category
def generate_summary(df, df_combined):
    """
    Generate product summaries and a comparison for top 3 products.
    
    Args:
        df (pd.DataFrame): Top products dataframe with columns: asin, product_name, category, 
                          review_count, avg_rating, avg_sentiment, recommend_ratio, helpfulness, composite_score
        df_combined (pd.DataFrame): Combined reviews dataframe containing 'asin' and 'cleaned_text' columns
    
    Returns:
        str: A formatted summary with individual product summaries and a comparative conclusion.
    """
    summaries = []
    comparison_data = []

    for idx, row in df.iterrows():
        asin = row['asin']
        product_name = row['product_name']
        avg_score = row['avg_rating']
        review_count = row['review_count']
        avg_sentiment = row['avg_sentiment']
        recommend_ratio = row['recommend_ratio']
        helpfulness = row['helpfulness']
        
        # Get reviews for this specific product from df_combined
        product_reviews = df_combined[df_combined['asins'] == asin]['cleaned_text'].tolist()
        
        # Handle case where no reviews are found
        if not product_reviews:
            summary_text = f"No reviews found for {product_name} (ASIN: {asin})"
            summaries.append(summary_text)
            continue
        
        # Limit reviews to avoid token limits (take first 10 reviews or combine if short)
        if len(product_reviews) > 10:
            formatted_reviews = "\n\n".join(product_reviews[:10])
            review_note = f"(Showing first 10 out of {len(product_reviews)} reviews)"
        else:
            formatted_reviews = "\n\n".join(product_reviews)
            review_note = f"(All {len(product_reviews)} reviews included)"
        
        # Convert sentiment score to descriptive text
        if avg_sentiment >= 0.6:
            sentiment_desc = "very positive"
        elif avg_sentiment >= 0.2:
            sentiment_desc = "positive"
        elif avg_sentiment >= -0.2:
            sentiment_desc = "neutral"
        elif avg_sentiment >= -0.6:
            sentiment_desc = "negative"
        else:
            sentiment_desc = "very negative"

        prompt = f"""
        You are an expert product reviewer.
        Summarize the following customer reviews for the product: '{product_name}'
        Write a maximum 5 lines summary, very tech-oriented.

        Product Details:
        - Product name: {product_name}
        - Average rating: {avg_score:.2f}/5.0 ({review_count} reviews)
        - Average sentiment: {sentiment_desc} ({avg_sentiment:.2f})
        - Recommendation ratio: {recommend_ratio:.1%}
        - Review helpfulness: {helpfulness:.2f}
        
        Customer Reviews {review_note}:
        {formatted_reviews}
        
        Focus on technical aspects, build quality, performance, and value proposition.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a technical product analyst specializing in portable electronics. Focus on technical specifications, performance metrics, and practical user experience."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3  # Lower temperature for more consistent technical summaries
            )
            summary_text = response.choices[0].message.content.strip()
        except Exception as e:
            summary_text = f"Error summarizing {product_name}: {e}"

        summaries.append(f"**{product_name}**\n{summary_text}")
        
        # Store comparison data
        comparison_data.append({
            "name": product_name,
            "rating": avg_score,
            "reviews": review_count,
            "sentiment": avg_sentiment,
            "recommend": recommend_ratio,
            "helpfulness": helpfulness,
            "composite": row['composite_score']
        })

    # Create a detailed comparison summary
    comparison_summary = "\n\n## Product Comparison\n"
    comparison_summary += "| Product | Rating | Reviews | Sentiment | Recommend % | Helpfulness | Composite |\n"
    comparison_summary += "|---------|--------|---------|-----------|-------------|-------------|----------|\n"
    
    for product in comparison_data:
        comparison_summary += (
            f"| {product['name'][:30]}{'...' if len(product['name']) > 30 else ''} | "
            f"{product['rating']:.2f} | {product['reviews']} | {product['sentiment']:.2f} | "
            f"{product['recommend']:.1%} | {product['helpfulness']:.2f} | {product['composite']:.2f} |\n"
        )
    
    # Add key insights
    best_rated = max(comparison_data, key=lambda x: x['rating'])
    most_reviewed = max(comparison_data, key=lambda x: x['reviews'])
    highest_composite = max(comparison_data, key=lambda x: x['composite'])
    
    insights = f"\n\n## Key Insights\n"
    insights += f"- **Highest Rated**: {best_rated['name']} ({best_rated['rating']:.2f}/5.0)\n"
    insights += f"- **Most Reviewed**: {most_reviewed['name']} ({most_reviewed['reviews']} reviews)\n"
    insights += f"- **Best Overall Score**: {highest_composite['name']} (composite: {highest_composite['composite']:.2f})\n"

    return "\n\n".join(summaries) + comparison_summary + insights
# Test the summary generation function
summary_text =generate_summary(top_products, df_combined)
print("\nGenerated Summary:")
print(summary_text)


# We then put all the models together in a single function that takes as input a user query (in the shape of one of the six meta-categories) 
# and returns the top 3 products for that meta-category, including the product name, image URL, aggregated rating, and a short summary of the reviews
# finally, we deploy the function as a web service using Gradio
