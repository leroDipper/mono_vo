import json
import numpy as np

def scale_cameras_sfm(input_path, output_path, scale_factor):
    # Load the SfM file
    with open(input_path, 'r') as f:
        sfm = json.load(f)
    
    # Scale all camera positions
    for pose in sfm["poses"]:
        center = pose["pose"]["transform"]["center"]
        # Convert to numpy, scale, convert back to list
        center_scaled = (np.array(center, dtype=np.float64) * scale_factor).tolist()
        pose["pose"]["transform"]["center"] = center_scaled
    
    # Save scaled version
    with open(output_path, 'w') as f:
        json.dump(sfm, f, indent=2)
    
    print(f"Scaled cameras saved to {output_path}")

if __name__ == "__main__":
    scale_factor = 0.2127
    input_file = "StructureFromMotion/output/cameras.sfm"
    output_file = "StructureFromMotion/output/cameras_scaled.sfm"
    
    scale_cameras_sfm(input_file, output_file, scale_factor)