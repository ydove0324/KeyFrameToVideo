import json
import numpy as np
import random
import os
import shutil
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

MIN_FLOW = 7
MAX_FLOW = 32

class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

def process_folder(args_tuple):
    """Process a single folder: copy and scan for mp4 files"""
    idx, src_folder, filtered_source_dir, key_frames_options, counter, total = args_tuple
    
    try:
        folder_name = os.path.basename(src_folder)
        dst_folder = os.path.join(filtered_source_dir, folder_name)
        
        # Copy folder
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.copytree(src_folder, dst_folder, symlinks=True)
        
        # Find all mp4 files recursively
        folder_metadata = []
        for root, dirs, files in os.walk(dst_folder):
            for file in files:
                if file.endswith('.mp4'):
                    # Create metadata entry
                    metadata_entry = {
                        "file_name": os.path.join(dst_folder, file),
                        "caption": "",
                        "key_frames_indices": random.choice(key_frames_options)
                    }
                    folder_metadata.append(metadata_entry)
        
        # Update progress
        current = counter.increment()
        if current % 10 == 0:  # Update every 10 folders to reduce output noise
            print(f"Processed {current}/{total} folders")
        
        return folder_metadata, None
        
    except Exception as e:
        error_msg = f"Error processing folder {src_folder}: {str(e)}"
        print(error_msg)
        return [], error_msg

def main():
    parser = argparse.ArgumentParser(description='Filter video data based on optical flow statistics')
    parser.add_argument('--stat_json', 
                       default="/share/project/lzx/video_data/pexels/pexels_stat.jsonl",
                       help='Path to the JSONL file containing flow statistics')
    parser.add_argument('--source_dir', 
                       default="/share/project/huangxu/video-data/pexel-clips-part2_1/",
                       help='Source directory containing video clip folders')
    parser.add_argument('--num_workers', 
                       type=int, 
                       default=16,
                       help='Number of worker threads for parallel processing')
    
    args = parser.parse_args()
    
    stat_json = args.stat_json
    source_dir = args.source_dir.rstrip('/') + '/'
    num_workers = args.num_workers
    
    # Load data and filter
    mean_flows = []
    new_folder_paths = []
    
    print(f"Loading data from: {stat_json}")
    
    with open(stat_json, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            mean_flow = item["flow_data"]["mean_flow"]
            path = item["path"]
            basename = os.path.splitext(os.path.basename(path))[0]  # Remove extension
            
            folder_path = os.path.join(source_dir, basename)
            if not os.path.exists(folder_path):
                continue
                
            mean_flows.append(mean_flow)
            new_folder_paths.append(folder_path)
    
    print(f"Found {len(mean_flows)} valid folders")
    
    # Filter by optical flow score: keep 7 <= mean_flow <= 32
    target_indices = []
    for i, mean_flow in enumerate(mean_flows):
        if MIN_FLOW <= mean_flow <= MAX_FLOW:
            target_indices.append(i)
    
    print(f"Selected {len(target_indices)} folders with optical flow scores between 7 and 32")
    
    # Print simple statistics
    selected_flows = [mean_flows[i] for i in target_indices]
    print(f"Mean flow range: {min(selected_flows):.6f} - {max(selected_flows):.6f}")
    print(f"Mean flow average: {np.mean(selected_flows):.6f}")
    
    # Copy folders to filtered directory
    filtered_source_dir = source_dir.rstrip('/') + '_filtered'
    os.makedirs(filtered_source_dir, exist_ok=True)
    
    print(f"Copying folders to: {filtered_source_dir}")
    print(f"Using {num_workers} worker threads")
    
    metadata_list = []
    key_frames_options = [[0, 16], [0, 8, 16], [0, 4, 8, 12, 16]]
    counter = ThreadSafeCounter()
    
    # Prepare tasks for parallel processing
    tasks = []
    for idx in target_indices:
        src_folder = new_folder_paths[idx]
        task_args = (idx, src_folder, filtered_source_dir, key_frames_options, counter, len(target_indices))
        tasks.append(task_args)
    
    # Process folders in parallel
    errors = []
    print("Starting parallel processing...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_folder, task): task for task in tasks}
        
        # Collect results
        for future in as_completed(future_to_task):
            folder_metadata, error = future.result()
            if error:
                errors.append(error)
            else:
                metadata_list.extend(folder_metadata)
    
    print(f"Completed processing all folders")
    
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    # Write metadata.jsonl
    metadata_path = os.path.join(filtered_source_dir, "metadata.jsonl")
    with open(metadata_path, 'w') as f:
        for metadata in metadata_list:
            f.write(json.dumps(metadata) + '\n')
    
    print(f"Created metadata.jsonl with {len(metadata_list)} entries")
    print(f"Filtered dataset created in: {filtered_source_dir}")

if __name__ == "__main__":
    main()
