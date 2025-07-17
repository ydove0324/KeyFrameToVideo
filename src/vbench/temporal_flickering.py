import numpy as np
from tqdm import tqdm
import cv2
from .utils import load_dimension_info,load_video_list,save_json

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
    dist_init,
)


def get_frames(video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frames.append(frame)
            else:
                break
        video.release()
        assert frames != []
        return frames


def mae_seq(frames):
    ssds = []
    for i in range(len(frames)-1):
        ssds.append(calculate_mae(frames[i], frames[i+1]))
    return np.array(ssds)


def calculate_mae(img1, img2):
    """Computing the mean absolute error (MAE) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))


def cal_score(video_path):
    """please ensure the video is static"""
    frames = get_frames(video_path)
    score_seq = mae_seq(frames)
    return (255.0 - np.mean(score_seq).item())/255.0


def temporal_flickering(video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        try:
            score_per_video = cal_score(video_path)
        except AssertionError:
            continue
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results


def compute_temporal_flickering(video_dir, device, submodules_list, **kwargs):
    video_list = load_video_list(video_dir)
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = temporal_flickering(video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute temporal flickering metric (supports distributed execution)")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory that contains videos to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use, e.g. cuda or cpu")
    args = parser.parse_args()

    # initialize torch.distributed (expects env vars provided by torchrun)
    dist_init()

    # compute
    all_results, video_results = compute_temporal_flickering(args.video_dir, args.device, {})

    # only let rank 0 write / print results
    if get_rank() == 0:
        print("Temporal Flickering Score:", all_results)
        result_json = {
            "video_dir": args.video_dir,
            "all_results": all_results,
            "video_results": video_results,
        }
        save_json(result_json, "temporal_flickering_results.json")






