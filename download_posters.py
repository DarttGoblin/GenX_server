import pandas as pd
import requests
import os

# Load CSV with movie poster URLs
csv_file = '../GenX.csv'
df = pd.read_csv(csv_file)

# Directory to save the downloaded images
output_dir = 'downloaded_posters'
os.makedirs(output_dir, exist_ok=True)

# Loop through the CSV rows and download each image
for index, row in df.iterrows():
    image_url = row['Poster_Url']  # Replace with the actual column name for URLs
    image_name = f"{index + 1}.jpg"
    image_path = os.path.join(output_dir, image_name)

    if os.path.exists(image_path):
        print(f"Skipping {image_name}, already exists.")
        continue

    try:
        # Send GET request to download the image
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Write the image content to a file
        with open(image_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {image_name}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {image_url}: {e}")
    