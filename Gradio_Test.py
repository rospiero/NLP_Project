import pandas as pd
import gradio as gr

# Load CSV once at startup
df = pd.read_csv("output/test.csv")

# Get unique category clusters
category_clusters = sorted(df['category_cluster'].unique())

# Function to generate product cards for a selected category
def show_products(selected_category):
    filtered_df = df[df['category_cluster'] == selected_category]

    # Build list of product cards as markdown
    cards = ""
    for _, row in filtered_df.iterrows():
        product_name = row['name']
        summary = row['summary']
        rating = row['avg_positive_rating']
        image_url = row['image_url']

        card = f"""
### {product_name}

![Product Image]({image_url})

**Average Positive Rating:** {rating} ⭐  
**Summary:**  
{summary}

---
"""
        cards += card

    return cards

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🛒 Amazon Product Summary Explorer")
    gr.Markdown("Select a product category to view top products with summaries, ratings, and images.")

    category_input = gr.Dropdown(category_clusters, label="Choose Category Cluster")
    output = gr.Markdown()

    category_input.change(show_products, inputs=category_input, outputs=output)

demo.launch(share=True)
