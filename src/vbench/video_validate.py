'''
torchrun --nnodes 1 --node_rank 0 --nproc_per_node 8 \
--master_port 29500 \
-m vbench.video_validate
'''
from .utils import init_submodules, save_json
from .temporal_flickering import compute_temporal_flickering
from .imaging_quality import compute_imaging_quality
# from .subject_consistency import compute_subject_consistency
from .aesthetic_quality import compute_aesthetic_quality
from .dynamic_degree import compute_dynamic_degree
from .motion_smoothness import compute_motion_smoothness
from .background_consistency import compute_background_consistency
from .distributed import dist_init, destroy_process_group
import os

def validate(video_dir,output_json_path):
    submoules_dict = init_submodules(['temporal_flickering','imaging_quality','aesthetic_quality','dynamic_degree','motion_smoothness','background_consistency'])
    result_json = {"video_dir":video_dir}
    all_results, _ = compute_temporal_flickering(video_dir, "cuda",submoules_dict['temporal_flickering'])
    result_json["temporal_flickering"] = all_results
    all_results, _ = compute_imaging_quality(video_dir, "cuda",submoules_dict['imaging_quality'])
    result_json["imaging_quality"] = all_results
    all_results, _ = compute_aesthetic_quality(video_dir, "cuda",submoules_dict['aesthetic_quality'])
    result_json["aesthetic_quality"] = all_results
    # all_results, _ = compute_subject_consistency(video_dir, "cuda",submoules_dict['subject_consistency'])
    # result_json["subject_consistency"] = all_results
    all_results, _ = compute_dynamic_degree(video_dir, "cuda",submoules_dict['dynamic_degree'])
    result_json["dynamic_degree"] = all_results
    all_results, _ = compute_motion_smoothness(video_dir, "cuda",submoules_dict['motion_smoothness'])
    result_json["motion_smoothness"] = all_results
    all_results, _ = compute_background_consistency(video_dir, "cuda",submoules_dict['background_consistency'])
    result_json["background_consistency"] = all_results
    save_json(result_json, output_json_path)
if __name__ == "__main__":
    dist_init()
    submoules_dict = init_submodules(['aesthetic_quality','dynamic_degree'])
    # Iterate from 6250 down to 4250 with step -250 (6250, 6000, ..., 4250)
    for frames in range(10000, 8500, -500):
        # Zero-pad to 5 digits (e.g. 6250 -> 06250)
        frames_str = f"{frames:05d}"

        video_dir = f"/share/project/huangxu/workspace/KeyFrameToVideo/validation_results/video-reconstruction/pexel6-video-{frames_str}"
        output_json_path = f"/share/project/huangxu/workspace/KeyFrameToVideo/src/vbench/results/video-reconstruction/pexel6-video-{frames_str}.json"

        # Ensure result directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        print(f"\n>>> Processing {video_dir} → {output_json_path}")
        validate(video_dir, output_json_path)
    destroy_process_group()

