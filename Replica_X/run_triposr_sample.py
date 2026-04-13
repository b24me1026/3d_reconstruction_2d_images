import os
import subprocess

print("Running Replica_X perfect 3D reconstruction on a sample image...")

# Define paths
image_path = os.path.abspath("input_images/chair.png")
output_dir = os.path.abspath("triposr_sample_output")

if not os.path.exists(image_path):
    print(f"Error: Sample image not found at {image_path}")
    exit(1)

# Execute the Replica_X run.py script
cmd = [
    "python", "run.py",
    image_path,
    "--output-dir", output_dir
]

print(f"Executing: {' '.join(cmd)}")
subprocess.run(cmd, check=True)

print(f"\nDone! Replica_X 3D Reconstruction saved to: {output_dir}")
