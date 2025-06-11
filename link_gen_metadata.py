#!/usr/bin/env python3
import os
import json
from pathlib import Path

# 输入文件：每行一个 JSON 字符串或直接是路径字符串
INPUT_LIST = 'pexel/part2_path_list_clean.jsonl'
# 输出目录
DEST_DIR = Path('./pexel_part')
# metadata 输出文件
METADATA_FILE = 'pexel_part/metadata.jsonl'

def main():
    # 1. 确保目标目录存在
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 读取路径列表
    paths = []
    max_idx = 20
    with open(INPUT_LIST, 'r', encoding='utf-8') as f:
        for line in f:
            max_idx -= 1
            if max_idx < 0:
                break
            line = line.strip()
            if not line:
                continue
            # 如果每行是 JSON 对象，比如 {"path": "..."}，可以这样解析：
            try:
                obj = json.loads(line)
                # 假设键是 "path"
                orig_path = obj.get('path', '')
                if not orig_path:
                    # 如果加载不是 JSON，直接当作路径
                    raise ValueError
            except:
                # 不是 JSON，直接当做纯路径字符串
                orig_path = line.strip('"')
            paths.append(orig_path)

    # 3. 为每个文件创建软链接，并收集 metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as meta_f:
        for orig in paths:
            src = Path(orig)
            if not src.exists():
                print(f'Warning: 源文件不存在，跳过: {src}')
                continue

            # 目标链接路径
            dest_link = DEST_DIR / src.name
            # 如果已经存在同名链接或文件，先移除
            if dest_link.exists() or dest_link.is_symlink():
                dest_link.unlink()

            # 创建软链接
            os.symlink(src.resolve(), dest_link)
            print(f'Linked: {dest_link} -> {src.resolve()}')

            # 写入 metadata.jsonl
            meta = {
                "file_name": src.name,
                "caption": ""
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + '\n')

    print(f'\n完成！共处理 {len(paths)} 个文件， metadata 保存在 {METADATA_FILE}')

if __name__ == '__main__':
    main()
