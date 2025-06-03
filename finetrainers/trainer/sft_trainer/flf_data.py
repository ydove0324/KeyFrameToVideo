from typing import Any, Dict
import torch
from finetrainers.logging import get_logger

logger = get_logger()


class IterableFirstLastFrameDataset(torch.utils.data.IterableDataset):
    """
    Dataset wrapper that extracts first and last frames from video data
    for first-last-frame to video training.
    """
    
    def __init__(self, dataset: torch.utils.data.IterableDataset, min_frames: int = 3):
        super().__init__()
        self.dataset = dataset
        self.min_frames = min_frames  # Minimum frames needed (first + middle + last)
        
        logger.info("Initialized IterableFirstLastFrameDataset")

    def __iter__(self):
        logger.info("Starting IterableFirstLastFrameDataset")
        for data in iter(self.dataset):
            flf_data = self._extract_first_last_frames(data)
            if flf_data is not None:
                yield flf_data

    def load_state_dict(self, state_dict):
        if hasattr(self.dataset, 'load_state_dict'):
            self.dataset.load_state_dict(state_dict)

    def state_dict(self):
        if hasattr(self.dataset, 'state_dict'):
            return self.dataset.state_dict()
        return {}

    def _extract_first_last_frames(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract first and last frames from video data.
        
        Args:
            data: Dictionary containing video data
            
        Returns:
            Modified data dictionary with first_frame and last_frame fields,
            or None if video is too short
        """
        # Only process if video is present
        if "video" not in data:
            logger.warning("No video found in data, skipping")
            return None
            
        video = data["video"]  # Expected shape: [B, F, C, H, W]
        
        if video.ndim != 5:
            logger.warning(f"Expected 5D video tensor, got {video.ndim}D, skipping")
            return None
            
        num_frames = video.shape[1]  # F dimension
        
        # Skip videos that are too short
        if num_frames < self.min_frames:
            logger.warning(f"Video has {num_frames} frames, need at least {self.min_frames}, skipping")
            return None
        
        # Extract first and last frames
        first_frame = video[:, 0]   # [B, C, H, W]
        last_frame = video[:, -1]   # [B, C, H, W]
        
        # Create modified data dictionary
        flf_data = dict(data)
        flf_data["first_frame"] = first_frame
        flf_data["last_frame"] = last_frame
        flf_data["num_frames"] = num_frames
        
        return flf_data


class ValidationFirstLastFrameDataset(torch.utils.data.IterableDataset):
    """
    Validation dataset for first-last-frame to video generation.
    Expects data with first_frame_path and last_frame_path.
    """
    
    def __init__(self, dataset: torch.utils.data.IterableDataset):
        super().__init__()
        self.dataset = dataset
        logger.info("Initialized ValidationFirstLastFrameDataset")

    def __iter__(self):
        logger.info("Starting ValidationFirstLastFrameDataset")
        for data in iter(self.dataset):
            # Validation data should already have first_frame and last_frame
            # or first_frame_path and last_frame_path
            yield data

    def load_state_dict(self, state_dict):
        if hasattr(self.dataset, 'load_state_dict'):
            self.dataset.load_state_dict(state_dict)

    def state_dict(self):
        if hasattr(self.dataset, 'state_dict'):
            return self.dataset.state_dict()
        return {} 