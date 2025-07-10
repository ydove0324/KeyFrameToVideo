import os
import json
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import random

# Try to import video validation libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


def validate_video_file(video_path: str, method: str = "auto") -> Tuple[bool, str]:
    """
    验证视频文件是否有效
    
    Args:
        video_path: 视频文件路径
        method: 验证方法 ("auto", "cv2", "pyav", "ffprobe")
    
    Returns:
        (is_valid, error_message): 是否有效和错误信息
    """
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    if os.path.getsize(video_path) == 0:
        return False, "File is empty"
    
    if method == "auto":
        # 优先级: ffprobe > pyav > cv2
        if shutil.which("ffprobe"):
            return validate_with_ffprobe(video_path)
        elif HAS_PYAV:
            return validate_with_pyav(video_path)
        elif HAS_CV2:
            return validate_with_cv2(video_path)
        else:
            return validate_basic(video_path)
    elif method == "ffprobe":
        return validate_with_ffprobe(video_path)
    elif method == "pyav":
        return validate_with_pyav(video_path)
    elif method == "cv2":
        return validate_with_cv2(video_path)
    else:
        return validate_basic(video_path)


def validate_with_ffprobe(video_path: str) -> Tuple[bool, str]:
    """使用ffprobe验证视频文件"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", 
            "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            return False, f"ffprobe error: {result.stderr.strip()}"
        
        # 检查是否有视频流
        import json
        data = json.loads(result.stdout)
        
        if "streams" not in data:
            return False, "No streams found"
        
        has_video = any(stream.get("codec_type") == "video" for stream in data["streams"])
        if not has_video:
            return False, "No video stream found"
        
        return True, "Valid"
        
    except subprocess.TimeoutExpired:
        return False, "Validation timeout"
    except Exception as e:
        return False, f"ffprobe validation error: {str(e)}"


def validate_with_pyav(video_path: str) -> Tuple[bool, str]:
    """使用PyAV验证视频文件"""
    if not HAS_PYAV:
        return False, "PyAV not available"
    
    try:
        container = av.open(video_path)
        
        # 检查是否有视频流
        video_streams = [s for s in container.streams if s.type == 'video']
        if not video_streams:
            container.close()
            return False, "No video streams found"
        
        # 尝试读取第一帧
        for frame in container.decode(video=0):
            break
        
        container.close()
        return True, "Valid"
        
    except av.error.InvalidDataError as e:
        return False, f"Invalid data: {str(e)}"
    except Exception as e:
        return False, f"PyAV validation error: {str(e)}"


def validate_with_cv2(video_path: str) -> Tuple[bool, str]:
    """使用OpenCV验证视频文件"""
    if not HAS_CV2:
        return False, "OpenCV not available"
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        # 检查视频属性
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return False, "No frames found"
        
        # 尝试读取第一帧
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False, "Cannot read first frame"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"OpenCV validation error: {str(e)}"


def validate_basic(video_path: str) -> Tuple[bool, str]:
    """基础验证（仅检查文件头）"""
    try:
        with open(video_path, 'rb') as f:
            # 检查MP4文件签名
            header = f.read(12)
            if len(header) < 12:
                return False, "File too short"
            
            # MP4文件应该在偏移4处有 'ftyp' 标识
            if header[4:8] == b'ftyp':
                return True, "Basic validation passed"
            else:
                return False, "Invalid MP4 header"
                
    except Exception as e:
        return False, f"Basic validation error: {str(e)}"


def find_mp4_files(directory: str, validate: bool = False, validation_method: str = "auto", delete_invalid: bool = False, confirm_delete: bool = True) -> Tuple[List[str], List[str]]:
    """
    递归查找指定目录下的所有mp4文件
    
    Args:
        directory: 要搜索的目录路径
        validate: 是否验证视频文件有效性
        validation_method: 验证方法
        delete_invalid: 是否删除无效文件
        confirm_delete: 是否在删除前确认
    
    Returns:
        (valid_files, invalid_files): 有效文件列表和无效文件列表
    """
    mp4_files = []
    directory_path = Path(directory)
    
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    print("Searching for MP4 files...")
    
    # 递归查找所有mp4文件
    for mp4_file in tqdm(list(directory_path.rglob("*.mp4")), desc="Scanning .mp4 files"):
        if mp4_file.is_file():
            mp4_files.append(str(mp4_file))
    
    # 也查找MP4大写扩展名
    for mp4_file in tqdm(list(directory_path.rglob("*.MP4")), desc="Scanning .MP4 files"):
        if mp4_file.is_file():
            mp4_files.append(str(mp4_file))
    
    mp4_files = sorted(mp4_files)
    
    if not validate:
        return mp4_files, []
    
    # 验证视频文件
    print(f"Validating {len(mp4_files)} video files...")
    valid_files = []
    invalid_files = []
    
    with tqdm(mp4_files, desc="Validating videos", unit="file") as pbar:
        for video_path in pbar:
            pbar.set_postfix_str(f"Current: {os.path.basename(video_path)[:30]}...")
            
            is_valid, error_msg = validate_video_file(video_path, validation_method)
            if is_valid:
                valid_files.append(video_path)
            else:
                invalid_files.append(video_path)
                print(f"\n❌ Invalid: {video_path} - {error_msg}")
    
    print(f"\n✅ Valid files: {len(valid_files)}")
    print(f"❌ Invalid files: {len(invalid_files)}")
    
    # 删除无效文件
    if delete_invalid and invalid_files:
        deleted_files = delete_invalid_files(invalid_files, confirm_delete)
        print(f"🗑️  Deleted {len(deleted_files)} invalid files")
        # 更新invalid_files列表，移除已删除的文件
        invalid_files = [f for f in invalid_files if f not in deleted_files]
    
    return valid_files, invalid_files


def delete_invalid_files(invalid_files: List[str], confirm_delete: bool = True) -> List[str]:
    """
    删除无效文件
    
    Args:
        invalid_files: 无效文件列表
        confirm_delete: 是否在删除前确认
    
    Returns:
        已删除的文件列表
    """
    if not invalid_files:
        return []
    
    print(f"\n⚠️  Found {len(invalid_files)} invalid files:")
    for i, file_path in enumerate(invalid_files[:10], 1):  # 只显示前10个
        print(f"  {i}. {file_path}")
    
    if len(invalid_files) > 10:
        print(f"  ... and {len(invalid_files) - 10} more files")
    
    if confirm_delete:
        print(f"\n⚠️  WARNING: This will permanently delete {len(invalid_files)} invalid video files!")
        print("This action cannot be undone.")
        
        while True:
            response = input("\nDo you want to proceed? (yes/no/list): ").lower().strip()
            
            if response in ['yes', 'y']:
                break
            elif response in ['no', 'n']:
                print("Deletion cancelled.")
                return []
            elif response in ['list', 'l']:
                print("\nComplete list of files to be deleted:")
                for i, file_path in enumerate(invalid_files, 1):
                    print(f"  {i:3d}. {file_path}")
                continue
            else:
                print("Please enter 'yes', 'no', or 'list'")
    
    # 执行删除
    deleted_files = []
    failed_deletions = []
    
    print(f"\n🗑️  Deleting {len(invalid_files)} invalid files...")
    with tqdm(invalid_files, desc="Deleting files", unit="file") as pbar:
        for file_path in pbar:
            pbar.set_postfix_str(f"Deleting: {os.path.basename(file_path)[:30]}...")
            
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except Exception as e:
                failed_deletions.append((file_path, str(e)))
                print(f"\n❌ Failed to delete {file_path}: {e}")
    
    if failed_deletions:
        print(f"\n⚠️  Failed to delete {len(failed_deletions)} files:")
        for file_path, error in failed_deletions:
            print(f"  {file_path}: {error}")
    
    return deleted_files


def generate_metadata_entry(video_path: str, use_relative_path: bool = True, base_dir: str = None) -> Dict[str, Any]:
    """
    为单个视频文件生成metadata条目
    
    Args:
        video_path: 视频文件路径
        use_relative_path: 是否使用相对路径
        base_dir: 基础目录（用于计算相对路径）
    
    Returns:
        metadata字典
    """
    if use_relative_path and base_dir:
        # 计算相对路径
        rel_path = os.path.relpath(video_path, base_dir)
        file_path = rel_path
    else:
        file_path = video_path
    key_frames_options = [[0,16],[0,8,16],[0,4,8,12,16]]
    return {
        "file_name": file_path,
        "caption": "",  # 空的caption
        "key_frames_indices": random.choice(key_frames_options)
    }


def split_first_n_videos(
    input_directory: str,
    n: int,
    output_dir: str = "part1",
    copy_files: bool = True,
    use_relative_path: bool = True,
    validate_videos: bool = True,
    validation_method: str = "auto",
    delete_invalid: bool = False,
    confirm_delete: bool = True
) -> None:
    """
    提取前n个mp4文件到新文件夹并生成metadata.jsonl
    
    Args:
        input_directory: 包含mp4文件的输入目录
        n: 要提取的文件数量
        output_dir: 输出目录名称
        copy_files: 是否复制文件（True）还是只创建符号链接（False）
        use_relative_path: 是否在jsonl中使用相对路径
        validate_videos: 是否验证视频文件有效性
        validation_method: 验证方法
        delete_invalid: 是否删除无效文件
        confirm_delete: 是否在删除前确认
    """
    print(f"Searching for MP4 files in: {input_directory}")
    
    # 查找所有mp4文件并验证
    valid_files, invalid_files = find_mp4_files(
        input_directory, 
        validate_videos, 
        validation_method, 
        delete_invalid, 
        confirm_delete
    )
    
    if not valid_files:
        print("No valid MP4 files found in the specified directory.")
        return
    
    if invalid_files:
        print(f"⚠️  Found {len(invalid_files)} invalid video files that will be skipped.")
        
        # 保存无效文件列表
        invalid_log = Path(output_dir) / "invalid_files.txt"
        os.makedirs(os.path.dirname(invalid_log), exist_ok=True)
        with open(invalid_log, 'w', encoding='utf-8') as f:
            for invalid_file in invalid_files:
                f.write(f"{invalid_file}\n")
        print(f"📝 Invalid files list saved to: {invalid_log}")
    
    if n > len(valid_files):
        print(f"Warning: Requested {n} files, but only {len(valid_files)} valid files found. Using all available valid files.")
        n = len(valid_files)
    
    # 选择前n个有效文件
    selected_files = valid_files[:n]
    print(f"Selected first {len(selected_files)} valid MP4 files")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"Created output directory: {output_path.absolute()}")
    
    # 复制或链接文件
    copied_files = []
    with tqdm(selected_files, desc="Processing files", unit="file") as pbar:
        for video_path in pbar:
            video_file = Path(video_path)
            output_file = output_path / video_file.name
            
            # 更新进度条显示当前文件
            pbar.set_postfix_str(f"Current: {video_file.name[:30]}...")
            
            try:
                if copy_files:
                    shutil.copy2(video_path, output_file)
                else:
                    # 创建符号链接（在Linux/Mac上）
                    if output_file.exists():
                        output_file.unlink()
                    output_file.symlink_to(Path(video_path).absolute())
                
                copied_files.append(str(output_file))
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
    
    print(f"Successfully {'copied' if copy_files else 'linked'} {len(copied_files)} files")
    
    # 生成metadata.jsonl
    metadata_file = output_path / "metadata.jsonl"
    print(f"Generating metadata to: {metadata_file}")
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for video_path in copied_files:
            metadata_entry = generate_metadata_entry(
                video_path, 
                use_relative_path=use_relative_path,
                base_dir=str(output_path)
            )
            
            # 写入jsonl格式（每行一个JSON对象）
            json_line = json.dumps(metadata_entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n✅ Split completed successfully!")
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📊 Files processed: {len(copied_files)}")
    print(f"📝 Metadata file: {metadata_file}")
    if invalid_files:
        print(f"⚠️  Invalid files skipped: {len(invalid_files)}")


def generate_metadata_jsonl(
    input_directory: str,
    output_file: str = "metadata.jsonl",
    use_relative_path: bool = True,
    validate_videos: bool = False,
    validation_method: str = "auto",
    delete_invalid: bool = False,
    confirm_delete: bool = True
) -> None:
    """
    生成metadata.jsonl文件
    
    Args:
        input_directory: 包含mp4文件的输入目录
        output_file: 输出的jsonl文件路径
        use_relative_path: 是否在jsonl中使用相对路径
        validate_videos: 是否验证视频文件有效性
        validation_method: 验证方法
        delete_invalid: 是否删除无效文件
        confirm_delete: 是否在删除前确认
    """
    print(f"Searching for MP4 files in: {input_directory}")
    
    # 查找所有mp4文件
    valid_files, invalid_files = find_mp4_files(
        input_directory, 
        validate_videos, 
        validation_method, 
        delete_invalid, 
        confirm_delete
    )
    
    if not valid_files:
        print("No valid MP4 files found in the specified directory.")
        return
    
    if invalid_files and validate_videos:
        print(f"⚠️  Found {len(invalid_files)} invalid video files that will be skipped.")
        
        # 保存无效文件列表
        invalid_log = Path(output_file).parent / "invalid_files.txt"
        with open(invalid_log, 'w', encoding='utf-8') as f:
            for invalid_file in invalid_files:
                f.write(f"{invalid_file}\n")
        print(f"📝 Invalid files list saved to: {invalid_log}")
    
    print(f"Found {len(valid_files)} valid MP4 files")
    print(f"Generating metadata to: {output_file}")
    
    # 生成metadata条目并写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用更详细的进度条配置
        pbar = tqdm(
            valid_files, 
            desc="Processing videos", 
            unit="file",
            ncols=100,  # 进度条宽度
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for video_path in pbar:
            metadata_entry = generate_metadata_entry(
                video_path, 
                use_relative_path=use_relative_path,
                base_dir=input_directory
            )
            
            # 更新进度条描述显示当前处理的文件
            current_file = os.path.basename(video_path)
            pbar.set_postfix_str(f"Current: {current_file[:30]}...")
            
            # 写入jsonl格式（每行一个JSON对象）
            json_line = json.dumps(metadata_entry, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n✅ Metadata file generated: {output_file}")
    print(f"📊 Total entries: {len(valid_files)}")
    if invalid_files and validate_videos:
        print(f"⚠️  Invalid files skipped: {len(invalid_files)}")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl for diffusers training")
    parser.add_argument("input_dir", help="Directory containing MP4 files")
    parser.add_argument("-o", "--output", default="metadata.jsonl", 
                        help="Output jsonl file path (default: metadata.jsonl)")
    parser.add_argument("--absolute-path", action="store_true",
                        help="Use absolute paths instead of relative paths")
    parser.add_argument("--preview", action="store_true",
                        help="Preview found files without generating metadata")
    
    # 新增的分割功能参数
    parser.add_argument("--split-first", type=int, metavar="N",
                        help="Split first N videos to a separate directory")
    parser.add_argument("--split-output-dir", default="part1",
                        help="Output directory name for split videos (default: part1)")
    parser.add_argument("--symlink", action="store_true",
                        help="Create symbolic links instead of copying files when splitting")
    
    # 视频验证参数
    parser.add_argument("--validate", action="store_true",
                        help="Validate video files before processing")
    parser.add_argument("--validation-method", choices=["auto", "ffprobe", "pyav", "cv2", "basic"],
                        default="auto", help="Video validation method (default: auto)")
    
    # 删除无效文件参数
    parser.add_argument("--delete-invalid", action="store_true",
                        help="Delete invalid video files instead of skipping them")
    parser.add_argument("--force-delete", action="store_true",
                        help="Delete invalid files without confirmation (use with caution!)")
    
    args = parser.parse_args()
    
    # 如果启用删除功能，自动启用验证
    if args.delete_invalid:
        args.validate = True
    
    try:
        if args.split_first:
            # 分割前n个视频的功能
            split_first_n_videos(
                input_directory=args.input_dir,
                n=args.split_first,
                output_dir=args.split_output_dir,
                copy_files=not args.symlink,
                use_relative_path=not args.absolute_path,
                validate_videos=args.validate,
                validation_method=args.validation_method,
                delete_invalid=args.delete_invalid,
                confirm_delete=not args.force_delete
            )
        elif args.preview:
            # 预览模式：只显示找到的文件
            valid_files, invalid_files = find_mp4_files(
                args.input_dir, 
                args.validate, 
                args.validation_method, 
                args.delete_invalid, 
                not args.force_delete
            )
            print(f"Found {len(valid_files)} valid MP4 files:")
            for i, file_path in enumerate(valid_files, 1):
                print(f"{i:3d}. {file_path}")
            
            if invalid_files:
                print(f"\nFound {len(invalid_files)} invalid files:")
                for i, file_path in enumerate(invalid_files, 1):
                    print(f"{i:3d}. {file_path}")
        else:
            # 生成metadata.jsonl文件
            generate_metadata_jsonl(
                input_directory=args.input_dir,
                output_file=args.output,
                use_relative_path=not args.absolute_path,
                validate_videos=args.validate,
                validation_method=args.validation_method,
                delete_invalid=args.delete_invalid,
                confirm_delete=not args.force_delete
            )
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 