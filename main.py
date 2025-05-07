from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
from typing import Dict, Any
import random

app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Read the uploaded image
    image_contents = await file.read()
    nparr = np.frombuffer(image_contents, np.uint8)
    
    try:
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")
    
    # Step 1: Grayscale + Gaussian Blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 2: Laplacian of Gaussian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Step 3: Binary Mask
    _, binary_mask = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    # Step 4: FAST Corner Detection
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(binary_mask, None)
    corner_positions = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
    
    # Step 5: Bottom-Left Most Corner
    min_x = min(corner_positions, key=lambda pos: pos[0])[0]
    extreme_left_corner = [pos for pos in corner_positions if pos[0] == min_x]
    botom_left_most_corner = extreme_left_corner[0]
    
    # Step 6: Top Edge Filtered Corners
    sorted_corners = sorted(corner_positions, key=lambda pos: pos[1])
    min_y = sorted_corners[0][1]
    filtered_corners = [pos for pos in sorted_corners if pos[1] <= min_y + 10]
    
    filtered_corners = [pos for pos in sorted_corners if pos[1] <= min_y + 9]
    top_left_most_corner = min(filtered_corners, key=lambda pos: pos[0])
    
    # Step 9: Angle Calculation
    x1, y1 = botom_left_most_corner
    x2, y2 = top_left_most_corner
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    m2 = 0
    
    if m1 == float('inf'):
        angle = 90
    else:
        angle_radians = np.arctan(abs((m1 - m2) / (1 + m1 * m2)))
        angle = np.degrees(angle_radians)
        angle_of_declination = 180 - angle
    
    # Process bottom edge corners
    bottom_left_y = botom_left_most_corner[1]
    y_tolerance = 10
    bottom_edge_corners = [pos for pos in corner_positions if abs(pos[1] - bottom_left_y) <= y_tolerance]
    
    # Filter clustered bottom-edge corners
    height, width = image.shape[:2]
    center_x = width // 2
    center_margin = 40
    
    left_zone = [pt for pt in bottom_edge_corners if pt[0] < center_x - center_margin]
    right_zone = [pt for pt in bottom_edge_corners if pt[0] > center_x + center_margin]
    center_zone = [pt for pt in bottom_edge_corners if center_x - center_margin <= pt[0] <= center_x + center_margin]
    
    # Left zone filtering
    left_filtered = []
    if left_zone:
        min_x = min(pt[0] for pt in left_zone)
        left_filtered = [pt for pt in left_zone if pt[0] == min_x]
        left_filtered = [max(left_filtered, key=lambda pt: pt[1])]
    
    # Right zone filtering
    right_filtered = []
    if right_zone:
        max_x = max(pt[0] for pt in right_zone)
        right_filtered = [pt for pt in right_zone if pt[0] == max_x]
        right_filtered = [max(right_filtered, key=lambda pt: pt[1])]
    
    # Center zone filtering
    center_filtered = []
    if center_zone:
        center_filtered = [max(center_zone, key=lambda pt: pt[1])]
    
    filtered_bottom_corners = left_filtered + center_filtered + right_filtered
    
    # Process top edge corners with zone-based deduplication
    top_edge_corners = [pos for pos in sorted_corners if pos[1] <= min_y + 10]
    
    left_zone = [pt for pt in top_edge_corners if pt[0] < center_x - center_margin]
    right_zone = [pt for pt in top_edge_corners if pt[0] > center_x + center_margin]
    center_zone = [pt for pt in top_edge_corners if center_x - center_margin <= pt[0] <= center_x + center_margin]
    
    # Left zone filtering
    left_filtered = []
    if left_zone:
        min_x = min(pt[0] for pt in left_zone)
        candidates = [pt for pt in left_zone if pt[0] == min_x]
        left_filtered = [min(candidates, key=lambda pt: pt[1])]
    
    # Right zone filtering
    right_filtered = []
    if right_zone:
        max_x = max(pt[0] for pt in right_zone)
        candidates = [pt for pt in right_zone if pt[0] == max_x]
        right_filtered = [min(candidates, key=lambda pt: pt[1])]
    
    # Center zone filtering
    center_filtered = []
    if len(center_zone) >= 5:
        def euclidean(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        is_cluster = all(euclidean(p1, p2) < 25 for i, p1 in enumerate(center_zone) 
                         for j, p2 in enumerate(center_zone) if i != j)
        
        if is_cluster:
            center_xs = [pt[0] for pt in center_zone]
            center_ys = [pt[1] for pt in center_zone]
            avg_x = int(np.mean(center_xs))
            avg_y = int(np.mean(center_ys))
            center_filtered = [min(center_zone, key=lambda pt: euclidean(pt, (avg_x, avg_y)))]
    
    filtered_top_corners = left_filtered + center_filtered + right_filtered
    
    # Combine filtered corners
    all_filtered_corners = filtered_top_corners + filtered_bottom_corners
    all_filtered_corners = list(set([tuple(map(int, pt)) for pt in all_filtered_corners]))
    
    # Manhattan distance function
    def manhattan_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    # Draw lines for visualization and count
    drawn_lines = set()
    line_count = 0
    
    # Connect each point to its two nearest neighbors by Manhattan distance
    for i, pt1 in enumerate(all_filtered_corners):
        distances = []
        
        for j, pt2 in enumerate(all_filtered_corners):
            if pt1 == pt2:
                continue
            
            dist = manhattan_dist(pt1, pt2)
            distances.append((dist, pt2))
        
        # Sort and pick two closest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = [pt for _, pt in distances[:2]]
        
        for pt2 in nearest_neighbors:
            line_key = frozenset((pt1, pt2))
            if line_key not in drawn_lines:
                drawn_lines.add(line_key)
                line_count += 1
    
    # Return the requested values
    return {
        "line_count": line_count,
        "angle_of_inclination": float(f"{angle:.2f}"),
        "angle_of_declination": float(f"{(180-angle):.2f}")
    }

@app.get("/")
async def root():
    return {"message": "Image Processing API is running. Send a POST request to /process-image/ with an image file."}