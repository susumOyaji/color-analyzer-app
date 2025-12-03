from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def extract_colors_kmeans(image_file, num_colors=5, crop_rect=None):
    try:
        # Open the image
        img = Image.open(image_file)
        
        # Crop if coordinates are provided
        if crop_rect:
            # crop_rect is expected to be (x, y, width, height)
            # PIL crop expects (left, upper, right, lower)
            x, y, w, h = crop_rect
            img = img.crop((x, y, x + w, y + h))
        
        # Resize image to speed up processing
        img = img.resize((150, 150))
        
        # Convert image to RGB
        img = img.convert('RGB')
        
        # Convert to numpy array
        img_np = np.array(img)
        
        # Reshape to a list of pixels
        pixels = img_np.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        # Ask for more clusters to find accent colors
        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and counts
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = Counter(labels)
        
        # Sort colors by count
        sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Process all candidates
        candidates = []
        for index, count in sorted_colors:
            color = colors[index]
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            
            # Filter out very light colors (background)
            if r > 230 and g > 230 and b > 230:
                continue
            
            # Calculate saturation (HSV model)
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val
            
            if max_val == 0:
                saturation = 0
            else:
                saturation = delta / max_val
                
            candidates.append({
                'r': r, 'g': g, 'b': b,
                'count': count,
                'hex': f"#{r:02x}{g:02x}{b:02x}".upper(),
                'saturation': saturation
            })

        # Select top colors
        final_colors = candidates[:num_colors]
        
        # If we have enough candidates, try to inject an accent color
        # Look for a high-saturation color that isn't already in the top list
        if len(candidates) > num_colors:
            # Sort remaining candidates by saturation
            remaining = candidates[num_colors:]
            remaining.sort(key=lambda x: x['saturation'], reverse=True)
            
            best_accent = remaining[0]
            
            # If the accent is significantly more saturated than the last picked color, swap it
            # Threshold: saturation > 0.3 and count > 10 (to avoid noise)
            if best_accent['saturation'] > 0.3 and best_accent['count'] > 10:
                # Replace the last color (usually the least dominant of the top N) with the accent
                final_colors[-1] = best_accent
                
        return final_colors
            
    except Exception as e:
        print(f"Error extracting colors: {e}")
        return []

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    num_colors = int(request.form.get('num_colors', 5))
    
    # Check for crop coordinates
    crop_rect = None
    if 'crop_x' in request.form:
        try:
            x = int(float(request.form.get('crop_x')))
            y = int(float(request.form.get('crop_y')))
            w = int(float(request.form.get('crop_w')))
            h = int(float(request.form.get('crop_h')))
            crop_rect = (x, y, w, h)
        except ValueError:
            pass
    
    colors = extract_colors_kmeans(file, num_colors, crop_rect)
    
    return jsonify({'colors': colors})

@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    num_slices = int(request.form.get('num_slices', 10))
    colors_per_slice = int(request.form.get('colors_per_slice', 3))
    
    # Check for crop coordinates
    crop_rect = None
    if 'crop_x' in request.form:
        try:
            x = int(float(request.form.get('crop_x')))
            y = int(float(request.form.get('crop_y')))
            w = int(float(request.form.get('crop_w')))
            h = int(float(request.form.get('crop_h')))
            crop_rect = (x, y, w, h)
        except ValueError:
            pass
    
    # Open and crop image
    img = Image.open(file)
    if crop_rect:
        x, y, w, h = crop_rect
        img = img.crop((x, y, x + w, y + h))
    
    img = img.convert('RGB')
    width, height = img.size
    
    # Calculate slice width
    slice_width = width // num_slices
    
    slices_data = []
    for i in range(num_slices):
        # Define slice boundaries
        left = i * slice_width
        right = left + slice_width if i < num_slices - 1 else width
        
        # Crop slice
        slice_img = img.crop((left, 0, right, height))
        
        # Convert to numpy array
        slice_np = np.array(slice_img)
        pixels = slice_np.reshape(-1, 3)
        
        # Use KMeans to find dominant colors in this slice
        n_clusters = min(colors_per_slice + 2, len(pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = Counter(labels)
        
        sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        slice_colors = []
        for index, count in sorted_colors:
            color = colors[index]
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            
            # Filter out very light colors
            if r > 230 and g > 230 and b > 230:
                continue
            
            # Calculate saturation
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val
            saturation = delta / max_val if max_val > 0 else 0
            
            slice_colors.append({
                'r': r, 'g': g, 'b': b,
                'count': count,
                'hex': f"#{r:02x}{g:02x}{b:02x}".upper(),
                'saturation': saturation
            })
            
            if len(slice_colors) >= colors_per_slice:
                break
        
        slices_data.append({
            'position': i,
            'left': left,
            'right': right,
            'colors': slice_colors
        })
    
    return jsonify({'slices': slices_data})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
