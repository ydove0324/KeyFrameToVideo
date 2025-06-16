import os
import shutil
from pathlib import Path

# 源文件夹
src_dir = Path('pexel')
# 目标文件夹前缀
dst_prefix = 'pexel_part2_'
# 分成的份数
num_parts = 10

# 获取所有 .mp4 软链接（仅保留软链接，非真实文件）
symlinks = sorted([f for f in src_dir.iterdir() if f.suffix == '.mp4' and f.is_symlink()])

# 平均分配
chunk_size = (len(symlinks) + num_parts - 1) // num_parts  # 向上取整

for i in range(num_parts):
    part_dir = Path(f'{dst_prefix}{i}')
    part_dir.mkdir(exist_ok=True)

    start = i * chunk_size
    end = min(start + chunk_size, len(symlinks))
    for link in symlinks[start:end]:
        dst_link = part_dir / link.name
        if dst_link.exists():
            dst_link.unlink()
        # 创建新的软链接，指向原始目标
        target = os.readlink(link)
        os.symlink(target, dst_link)

print(f"已将 {len(symlinks)} 个软链接分成 {num_parts} 份。")
