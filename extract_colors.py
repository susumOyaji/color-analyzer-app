from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import os

def extract_colors(image_path, num_colors=5):
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Resize image to speed up processing
        img = img.resize((150, 150))
        
        # Convert image to RGB
        img = img.convert('RGB')
        
        # Convert to numpy array
        img_np = np.array(img)
        
        # Reshape to a list of pixels
        pixels = img_np.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=num_colors + 2, random_state=42, n_init=10) # Ask for more clusters to allow filtering
        kmeans.fit(pixels)
        
        # Get colors and counts
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Sort colors by count
        sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top colors for {image_path} (using KMeans):")
        
        final_colors = []
        for index, count in sorted_colors:
            color = colors[index]
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            
            # Filter out very light colors (background) if they are too dominant
            # Heuristic: if R, G, B are all > 230, it's likely white/off-white background
            if r > 230 and g > 230 and b > 230:
                continue
                
            final_colors.append((r, g, b, count))
            if len(final_colors) >= num_colors:
                break
        
        for r, g, b, count in final_colors:
            print(f"RGB: ({r}, {g}, {b}), Hex: #{r:02x}{g:02x}{b:02x}, Count: {count}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    image_path = r"C:\Users\kanda\OneDrive\Desktop\R-25102.jpg"
    if os.path.exists(image_path):
        extract_colors(image_path, num_colors=20)
    else:
        print(f"File not found: {image_path}")
