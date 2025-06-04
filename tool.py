import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def find_mp4_files(directory: str) -> List[str]:
    """
    递归查找指定目录下的所有mp4文件
    
    Args:
        directory: 要搜索的目录路径
    
    Returns:
        mp4文件路径列表
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
    
    return sorted(mp4_files)


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
    
    return {
        "file_name": file_path,
        "text": ""  # 空的caption
    }


def generate_metadata_jsonl(
    input_directory: str,
    output_file: str = "metadata.jsonl",
    use_relative_path: bool = True
) -> None:
    """
    生成metadata.jsonl文件
    
    Args:
        input_directory: 包含mp4文件的输入目录
        output_file: 输出的jsonl文件路径
        use_relative_path: 是否在jsonl中使用相对路径
    """
    print(f"Searching for MP4 files in: {input_directory}")
    
    # 查找所有mp4文件
    mp4_files = find_mp4_files(input_directory)
    
    if not mp4_files:
        print("No MP4 files found in the specified directory.")
        return
    
    print(f"Found {len(mp4_files)} MP4 files")
    print(f"Generating metadata to: {output_file}")
    
    # 生成metadata条目并写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用更详细的进度条配置
        pbar = tqdm(
            mp4_files, 
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
    print(f"📊 Total entries: {len(mp4_files)}")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl for diffusers training")
    parser.add_argument("input_dir", help="Directory containing MP4 files")
    parser.add_argument("-o", "--output", default="metadata.jsonl", 
                        help="Output jsonl file path (default: metadata.jsonl)")
    parser.add_argument("--absolute-path", action="store_true",
                        help="Use absolute paths instead of relative paths")
    parser.add_argument("--preview", action="store_true",
                        help="Preview found files without generating metadata")
    
    args = parser.parse_args()
    
    try:
        if args.preview:
            # 预览模式：只显示找到的文件
            mp4_files = find_mp4_files(args.input_dir)
            print(f"Found {len(mp4_files)} MP4 files:")
            for i, file_path in enumerate(mp4_files, 1):
                print(f"{i:3d}. {file_path}")
        else:
            # 生成metadata.jsonl文件
            generate_metadata_jsonl(
                input_directory=args.input_dir,
                output_file=args.output,
                use_relative_path=not args.absolute_path
            )
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 