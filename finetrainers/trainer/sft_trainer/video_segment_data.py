import json
import pathlib
from typing import Any, Dict, Iterator, Optional, Union

import torch
import torch.utils.data
import torch.distributed
from finetrainers import logging
import time

logger = logging.get_logger()


class IterableVideoSegmentDataset(torch.utils.data.IterableDataset, torch.distributed.checkpoint.stateful.Stateful):
    """
    Dataset that reads from metadata.jsonl and splits videos into segments.
    Each segment contains a specified number of frames (e.g., 17 or 9 frames).
    The dataset splits videos from head to tail and discards remaining frames 
    that cannot form a complete segment.
    
    Supports distributed training by automatically sharding data across ranks.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.IterableDataset,
        frames_per_segment: int = 17,
        overlap_frames: int = 0,
    ):
        """
        Initialize the video segment dataset.
        
        Args:
            base_dataset: The underlying dataset to wrap
            frames_per_segment: Number of frames per segment (default: 17)
            overlap_frames: Number of overlapping frames between segments (default: 0)
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.frames_per_segment = frames_per_segment
        self.overlap_frames = overlap_frames
        self._sample_index = 0
        
        if self.frames_per_segment <= 0:
            raise ValueError("frames_per_segment must be positive")
        if self.overlap_frames < 0:
            raise ValueError("overlap_frames must be non-negative")
        if self.overlap_frames >= self.frames_per_segment:
            raise ValueError("overlap_frames must be less than frames_per_segment")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through the dataset, yielding video segments.
        Supports distributed training by sharding based on rank.
        """
        # Get distributed info
        world_size = 1
        rank = 0
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        
        segment_count = 0
        for data in self.base_dataset:
            # Process each video and yield segments
            segments = self._segment_video_data(data)
            for segment in segments:
                # Distributed sharding: only yield segments assigned to this rank
                if segment_count % world_size == rank:
                    self._sample_index += 1
                    yield segment
                segment_count += 1

    def _segment_video_data(self, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Split video data into segments.
        
        Args:
            data: Dictionary containing video data
            
        Yields:
            Dictionary for each video segment
        """
        # Only process if video is present
        if "video" not in data:
            logger.warning("No video found in data, skipping")
            return
            
        video = data["video"]  # Expected shape: [B, F, C, H, W] or [F, C, H, W]
        
        # Handle different video tensor dimensions
        if video.ndim == 4:  # [F, C, H, W]
            num_frames = video.shape[0]
            batch_size = 1
        elif video.ndim == 5:  # [B, F, C, H, W]
            batch_size, num_frames = video.shape[:2]
        else:
            logger.warning(f"Expected 4D or 5D video tensor, got {video.ndim}D, skipping")
            return
        
        if num_frames < self.frames_per_segment:
            logger.warning(f"Video has {num_frames} frames, need at least {self.frames_per_segment}, skipping")
            return
        
        # Calculate step size between segments
        step_size = self.frames_per_segment - self.overlap_frames
        
        # Generate segments
        start_frame = 0
        segment_idx = 0
        
        while start_frame + self.frames_per_segment <= num_frames:
            end_frame = start_frame + self.frames_per_segment
            
            # Extract segment
            if video.ndim == 4:  # [F, C, H, W]
                video_segment = video[start_frame:end_frame]
            else:  # [B, F, C, H, W]
                video_segment = video[:, start_frame:end_frame]
            
            # Create segment data dictionary
            segment_data = dict(data)
            segment_data["video"] = video_segment
            segment_data["segment_info"] = {
                "segment_idx": segment_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "total_frames": num_frames,
                "frames_per_segment": self.frames_per_segment,
            }
            
            yield segment_data
            
            # Move to next segment
            start_frame += step_size
            segment_idx += 1
            
        logger.debug(f"Generated {segment_idx} segments from video with {num_frames} frames")

    def load_state_dict(self, state_dict):
        """Load state for checkpoint resumption."""
        self._sample_index = state_dict.get("sample_index", 0)
        if hasattr(self.base_dataset, 'load_state_dict'):
            self.base_dataset.load_state_dict(state_dict.get("base_dataset", {}))

    def state_dict(self):
        """Save state for checkpointing."""
        state = {"sample_index": self._sample_index}
        if hasattr(self.base_dataset, 'state_dict'):
            state["base_dataset"] = self.base_dataset.state_dict()
        return state


class ValidationVideoSegmentDataset(torch.utils.data.Dataset):
    """
    Validation dataset for video segmentation that reads from a metadata file.
    """

    def __init__(
        self,
        metadata_file: str,
        frames_per_segment: int = 17,
        overlap_frames: int = 0,
    ):
        """
        Initialize validation dataset.
        
        Args:
            metadata_file: Path to metadata.jsonl file
            frames_per_segment: Number of frames per segment (default: 17)
            overlap_frames: Number of overlapping frames between segments (default: 0)
        """
        super().__init__()
        self.metadata_file = pathlib.Path(metadata_file)
        self.frames_per_segment = frames_per_segment
        self.overlap_frames = overlap_frames
        
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file {self.metadata_file} does not exist")
        
        # Load metadata
        self.metadata = []
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.metadata)} items from {self.metadata_file}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a validation item.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing validation data
        """
        return self.metadata[idx] 