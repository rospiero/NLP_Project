import pandas as pd
import gradio as gr
from pathlib import Path
import urllib.parse  # Fix for missing import

# print current working directory to check path
print("Current working directory:", Path.cwd())

# Load CSV once at startup
path = Path.cwd() / "Product_Summaries" / "Summaries_Combined.csv"
df = pd.read_csv(path)
# Ensure necessary columns are present and drop the non-necessary columns
necessary_columns = ['asin', 'product_name', 'category', 'image_url', 'avg_rating', 'review_count', 'sentiment_description', 'summary', 'is_low_data_category']
df = df[necessary_columns]

# Extract the best possible image from the image_url column
def extract_best_image_url(url_string):
    """
    Extract the best valid image URL from a comma-separated list.
    Decodes URL-encoded parts, filters for known image hosts.
    """
    placeholder = "https://upload.wikimedia.org/wikipedia/commons/a/ac/No_image_available.svg"
    
    if pd.isna(url_string) or not url_string.strip():
        return placeholder

    urls = [url.strip() for url in url_string.split(",") if url.strip()]
    
    trusted_domains = [
        'amazon.com',
        'ebayimg.com'
    ]

    for url in urls:
        decoded_url = urllib.parse.unquote(url)
        for domain in trusted_domains:
            if domain in decoded_url:
                return decoded_url  # return first valid match

    # fallback if none matched
    return placeholder

# Fix: Get unique categories from 'category' column
category_clusters = sorted(df['category'].unique())

# Updated function to generate product cards for a selected category
def show_products(selected_category):
    filtered_df = df[df['category'] == selected_category]

    if filtered_df.empty:
        return "⚠ No products found for this category."

    cards = ""
    for _, row in filtered_df.iterrows():
        image_url = extract_best_image_url(row['image_url'])
        warning = "**⚠ Please note that there is little data available for this product.**\n\n" if row['is_low_data_category'] else ""

        card = f"""
### {row['product_name']}

<img src="{image_url}" alt="{row['product_name']}" width="300"/>

{warning}
**ASIN:** {row['asin']}  
**Average Rating:** {row['avg_rating']} ⭐  
**Review Count:** {row['review_count']}  
**Sentiment Summary:** {row['sentiment_description']}  
**Summary:**  
{row['summary']}

---
"""
        cards += card

    return cards

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🛒 Amazon Product Summary Explorer")
    gr.Markdown("Select a product category to view top products with summaries, ratings, and images.")

    category_input = gr.Dropdown(category_clusters, label="Choose Product Category")
    output = gr.Markdown()

    category_input.change(show_products, inputs=category_input, outputs=output)

demo.launch(share=True)
