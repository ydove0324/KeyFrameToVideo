import json
import os
import webdataset as wds
from pathlib import Path
from typing import List, Dict

# Paths to input metadata and output directory
base_dir = "/share/project/huangxu/video-data/pexel-clips-part2_0_filtered"
metadata_path = "metadata.jsonl"
output_dir = "/share/project/huangxu/video-data/pexel-clips-part2_0_filtered_webdataset"
os.makedirs(output_dir, exist_ok=True)

# Initialize WebDataset ShardWriter with sharding
sink = wds.ShardWriter(os.path.join(output_dir, "dataset_%06d.tar"), maxcount=1000)

def get_unique_key(file_path: str, base_dir: str) -> str:
    """Generate a unique key for each sample using its relative path.
    
    Args:
        file_path: The full path to the file
        base_dir: The base directory to calculate relative path from
    
    Returns:
        A unique key string with directory structure encoded
    """
    rel_path = os.path.relpath(file_path, base_dir)
    # Replace directory separators with underscores and remove extension
    key = os.path.splitext(rel_path)[0].replace(os.sep, '_')
    return key

def process_video(file_name: str, base_dir: str, caption: str, key_frames_indices: List[int]) -> Dict:
    """Process a single video file and return a WebDataset sample."""
    video_path = os.path.join(base_dir, file_name)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    sample = {
        "__key__": get_unique_key(video_path, base_dir),  # Use relative path as key
        "mp4": open(video_path, "rb").read(),  # Read video file as binary
        "txt": caption,  # Store caption as text
        "json": json.dumps({
            "key_frames_indices": key_frames_indices,
            "original_path": file_name,  # Store original path for reference
            "caption": caption
        })  # Store metadata as JSON
    }
    return sample

# Read metadata.jsonl
with open(os.path.join(base_dir, metadata_path), 'r', encoding='utf-8') as f:
    for line in f:
        # Parse JSON line
        item = json.loads(line.strip())
        file_name = item["file_name"]
        caption = item["caption"]
        key_frames_indices = item["key_frames_indices"]
        
        # Process video and get sample
        sample = process_video(file_name, base_dir, caption, key_frames_indices)
        
        # Write sample to tar
        sink.write(sample)

# Close the writer
sink.close()
print(f"WebDataset shards created in {output_dir}")