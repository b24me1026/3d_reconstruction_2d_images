import torch
import torch.nn as nn
import cv2
import open3d as o3d
import numpy as np
import os
import json

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),
            nn.ReLU(),
            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.ReLU(),
            nn.Conv2d(128,256,4,2,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*14*14,1024)
        )

    def forward(self,x):
        x = self.encoder(x)
        return x

class PointDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1024,2048),
            nn.ReLU(),
            nn.Linear(2048,1024*3)
        )

    def forward(self,x):
        x = self.decoder(x)
        x = x.view(-1,1024,3)
        return x

class ReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = PointDecoder()

    def forward(self,x):
        x = self.encoder(x)
        points = self.decoder(x)
        return points

def extract_category_from_path(model_path):
    parts = model_path.replace("\\\\", "/").split("/")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"

def main():
    print("Initializing bulk reconstruction pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ReconstructionModel().to(device)
    model_path = "pix3d_reconstruction_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset_json_path = "pix3d/pix3d.json"
    if not os.path.exists(dataset_json_path):
        print(f"Error: Dataset JSON not found at {dataset_json_path}.")
        return
        
    with open(dataset_json_path, "r") as f:
        pix3d_data = json.load(f)

    # Process the first 50 objects as a batch
    output_dir = "reconstructed_points_bulk"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting bulk reconstruction for multiple 3D objects...")
    count = 0
    max_objects = 50

    for idx, sample in enumerate(pix3d_data):
        if count >= max_objects:
            break
            
        img_path = os.path.join("pix3d", sample["img"])
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.resize(img,(224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.tensor(img).permute(2,0,1).float()/255
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            predicted_points = model(img_tensor)
            
        predicted_points = predicted_points.cpu().numpy()[0]
        
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(predicted_points)
        
        # Estimate normals
        pcd_pred.estimate_normals()
        
        # Determine category and write out
        category = sample.get("category", extract_category_from_path(sample.get("model", "")))
        out_filename = os.path.join(output_dir, f"obj_{idx}_{category}.ply")
        o3d.io.write_point_cloud(out_filename, pcd_pred)
        count += 1
        
        if count % 10 == 0:
            print(f"Processed {count}/{max_objects} objects...")
            
    print(f"Done! Successfully generated {count} 3D point cloud objects in '{output_dir}'.")

if __name__ == "__main__":
    main()
