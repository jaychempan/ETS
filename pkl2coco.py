import json
import pickle
import os
import torch
import numpy as np
from tqdm import tqdm

def convert_tensors_to_lists(obj):
    """递归转换 PyTorch Tensor 和 numpy 数组为 Python 列表，以便 JSON 序列化"""
    if isinstance(obj, dict):
        return {k: convert_tensors_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(v) for v in obj]
    elif isinstance(obj, (torch.Tensor, np.ndarray)):  # 处理 Tensor 和 numpy 数组
        return obj.tolist()
    else:
        return obj

def update_coco_annotations(
    coco_file_path: str,
    pkl_file_path: str,
    output_json_path: str,
    annotations_output_path: str,
    score_threshold: float = 0.5
):
    """更新 COCO 标注并保存新的 JSON 文件"""
    # 读取 COCO JSON 文件
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)

    # 读取 PKL 文件
    with open(pkl_file_path, 'rb') as f:
        pkl_data = pickle.load(f)

    # 构建 file_name 到 image_id 的映射
    file_name_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
    
    # 构建 category name 到 category_id 的映射
    category_name_to_id = {cat["name"]: cat["id"] for cat in coco_data["categories"]}

    # 读取 PKL 数据并更新 COCO annotations
    new_annotations = []
    ann_id = 1  # 重新编号 annotations ID

    for item in tqdm(pkl_data, desc='Updating annotations'):
        img_path = item["img_path"]
        file_name = os.path.basename(img_path)  # 提取文件名
        if file_name not in file_name_to_id:
            continue  # 跳过没有匹配的文件

        image_id = file_name_to_id[file_name]
        pred_instances = item.get("pred_instances", {})
        bboxes = pred_instances.get("bboxes", [])
        scores = pred_instances.get("scores", [])
        labels = pred_instances.get("label_names", [])  # 确保是 label_names

        for bbox, score, label in zip(bboxes, scores, labels):
            if score < score_threshold:
                continue  # 过滤低置信度目标
            
            if label not in category_name_to_id:
                continue  # 跳过未知类别
            
            category_id = category_name_to_id[label]
            
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            new_annotations.append({
                # "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,  # 使用 COCO categories 中的映射 ID
                "bbox": [x_min, y_min, width, height],
                # "area": width * height,
                # "iscrowd": 0,
                "score": score
            })
            ann_id += 1

    # 保存完整的 COCO JSON（包含更新后的 annotations）
    coco_data["annotations"] = new_annotations
    # print(new_annotations)
    with open(output_json_path, 'w') as f:
        json.dump(convert_tensors_to_lists(coco_data), f, indent=4)
    print(f"Updated COCO JSON saved to {output_json_path}")

    # 仅保存 annotations 到单独的 JSON 文件
    with open(annotations_output_path, 'w') as f:
        json.dump(convert_tensors_to_lists(new_annotations), f, indent=4)
    print(f"Annotations JSON saved to {annotations_output_path}")

# 示例调用
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update COCO annotations with PKL predictions")
    parser.add_argument("--coco_file", type=str, required=True, help="Path to COCO JSON file")
    parser.add_argument("--pkl_file", type=str, required=True, help="Path to PKL file with predictions")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save updated COCO JSON")
    parser.add_argument("--annotations_json", type=str, required=True, help="Path to save only updated annotations")
    parser.add_argument("--score_threshold", type=float, default=0.0, help="Confidence score threshold")
    
    args = parser.parse_args()
    
    update_coco_annotations(
        coco_file_path=args.coco_file,
        pkl_file_path=args.pkl_file,
        output_json_path=args.output_json,
        annotations_output_path=args.annotations_json,
        score_threshold=args.score_threshold
    )
