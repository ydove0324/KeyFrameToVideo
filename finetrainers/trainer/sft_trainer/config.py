import argparse
from typing import TYPE_CHECKING, Any, Dict, List, Union

from finetrainers.utils import ArgsConfigMixin


if TYPE_CHECKING:
    from finetrainers.args import BaseArgs


class SFTLowRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT low rank training.

    Args:
        rank (int):
            Rank of the low rank approximation matrix.
        lora_alpha (int):
            The lora_alpha parameter to compute scaling factor (lora_alpha / rank) for low-rank matrices.
        target_modules (`str` or `List[str]`):
            Target modules for the low rank approximation matrices. Can be a regex string or a list of regex strings.
    """

    rank: int = 64
    lora_alpha: int = 64
    target_modules: Union[str, List[str]] = "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules",
            type=str,
            nargs="+",
            default=["(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"],
        )

    def validate_args(self, args: "BaseArgs"):
        assert self.rank > 0, "Rank must be a positive integer."
        assert self.lora_alpha > 0, "lora_alpha must be a positive integer."

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.rank = argparse_args.rank
        mapped_args.lora_alpha = argparse_args.lora_alpha
        mapped_args.target_modules = (
            argparse_args.target_modules[0] if len(argparse_args.target_modules) == 1 else argparse_args.target_modules
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"rank": self.rank, "lora_alpha": self.lora_alpha, "target_modules": self.target_modules}


class SFTFullRankConfig(ArgsConfigMixin):
    r"""
    Configuration class for SFT full rank training.
    """

    def add_args(self, parser: argparse.ArgumentParser):
        pass

    def validate_args(self, args: "BaseArgs"):
        pass

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        pass


class SFTFirstLastFrameConfig(ArgsConfigMixin):
    r"""
    Configuration class for First-Last-Frame to Video training.
    
    Args:
        training_mode (str):
            Training mode, should be 'first_last_frame' for FLF training.
        min_frames (int):
            Minimum number of frames required in video data.
        use_image_conditioning (bool):
            Whether to use image conditioning for the first and last frames.
    """

    training_mode: str = "first_last_frame"
    min_frames: int = 3
    use_image_conditioning: bool = True

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--training_mode", type=str, default="first_last_frame", choices=["first_last_frame"])
        parser.add_argument("--min_frames", type=int, default=3)
        parser.add_argument("--use_image_conditioning", action="store_true")

    def validate_args(self, args: "BaseArgs"):
        assert self.min_frames >= 2, "min_frames must be at least 2 for first and last frame."

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.training_mode = argparse_args.training_mode
        mapped_args.min_frames = argparse_args.min_frames
        mapped_args.use_image_conditioning = argparse_args.use_image_conditioning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "training_mode": self.training_mode,
            "min_frames": self.min_frames,
            "use_image_conditioning": self.use_image_conditioning,
        }


class SFTVideoSegmentConfig(ArgsConfigMixin):
    r"""
    Configuration class for Video Segmentation training.
    
    Args:
        enable_video_segmentation (bool):
            Whether to enable video segmentation dataset.
        frames_per_segment (int):
            Number of frames per video segment (default: 17).
        overlap_frames (int):
            Number of overlapping frames between segments (default: 0).
    """

    enable_video_segmentation: bool = False
    frames_per_segment: int = 17
    overlap_frames: int = 0

    def add_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--enable_video_segmentation", action="store_true", 
                          help="Enable video segmentation dataset")
        parser.add_argument("--frames_per_segment", type=int, default=17,
                          help="Number of frames per video segment")
        parser.add_argument("--overlap_frames", type=int, default=0,
                          help="Number of overlapping frames between segments")

    def validate_args(self, args: "BaseArgs"):
        assert self.frames_per_segment > 0, "frames_per_segment must be positive"
        assert self.overlap_frames >= 0, "overlap_frames must be non-negative"
        assert self.overlap_frames < self.frames_per_segment, "overlap_frames must be less than frames_per_segment"

    def map_args(self, argparse_args: argparse.Namespace, mapped_args: "BaseArgs"):
        mapped_args.enable_video_segmentation = argparse_args.enable_video_segmentation
        mapped_args.frames_per_segment = argparse_args.frames_per_segment
        mapped_args.overlap_frames = argparse_args.overlap_frames

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_video_segmentation": self.enable_video_segmentation,
            "frames_per_segment": self.frames_per_segment,
            "overlap_frames": self.overlap_frames,
        }
