import os
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

EXP_PATTERN = re.compile(r"exp\d+_gpu\d+")

def process_single_exp(exp_path, exp_name):
    try:
        subdirs = [os.path.join(exp_path, sd) for sd in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, sd))]
        if len(subdirs) != 1:
            return None

        log_dir = subdirs[0]
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
        if not log_files:
            return None

        log_path = os.path.join(log_dir, log_files[0])
        best_map = -1
        best_epoch = -1
        best_line = ""

        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i in range(len(lines) - 1):
            line = lines[i]
            if 'bbox_mAP_copypaste:' in line:
                map_match = re.search(r'bbox_mAP_copypaste:\s+([0-9.]+)', line)
                next_line = lines[i + 1]
                epoch_match = re.search(r'Epoch\(val\)\s*\[(\d+)\]', next_line)

                if map_match:
                    current_map = float(map_match.group(1))
                    current_epoch = int(epoch_match.group(1)) if epoch_match else -1
                    if current_map > best_map:
                        best_map = current_map
                        best_epoch = current_epoch
                        best_line = line.strip()

        if best_map >= 0:
            return {
                "exp_name": exp_name,
                "epoch": best_epoch,
                "mAP": best_map,
                "log_line": best_line
            }
    except Exception as e:
        print(f"❌ 错误处理 {exp_name}: {e}")
        return None


def sort_key(exp_name):
    match = re.match(r'exp(\d+)_gpu(\d+)', exp_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (float('inf'), float('inf'))


def find_best_map_in_logs_multithread(root_dir, output_csv="best_map_summary.csv", max_workers=8):
    exp_dirs = [
        d for d in os.listdir(root_dir)
        if EXP_PATTERN.fullmatch(d) and os.path.isdir(os.path.join(root_dir, d))
    ]
    
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(process_single_exp, os.path.join(root_dir, exp), exp): exp
            for exp in exp_dirs
        }

        for future in as_completed(future_to_exp):
            result = future.result()
            if result:
                results.append(result)

    # ✅ 按 exp数字排序
    results.sort(key=lambda x: sort_key(x["exp_name"]))

    # 写入 CSV
    csv_path = os.path.join(root_dir, output_csv)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["exp_name", "epoch", "mAP", "log_line"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n✅ 多线程统计完成，已排序保存至：{csv_path}")

# 用法
root_directory = "/home/panjiancheng/projects/NTIRE2025-CDFSOD-AI4EarthLab/mmdetection/work_dirs"  # ← 替换为你的路径
find_best_map_in_logs_multithread(root_directory)
