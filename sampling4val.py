import json
import random
from collections import defaultdict
from pathlib import Path

def sample_coco_uniform_ratio(
    coco_file_path: str,
    sample_ratio: float,
    seed: int = 42
):
    random.seed(seed)

    # 解析路径
    coco_path = Path(coco_file_path)
    output_file_path = coco_path.parent / f"test_val_{sample_ratio}.json"

    # 读取 COCO 数据
    with open(coco_file_path, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']
    categories = coco['categories']

    # 构建索引
    category_to_image_ids = defaultdict(set)
    image_id_to_annotations = defaultdict(list)

    for ann in annotations:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        category_to_image_ids[cat_id].add(img_id)
        image_id_to_annotations[img_id].append(ann)

    # 采样图像ID
    selected_image_ids = set()
    for cat_id, image_ids in category_to_image_ids.items():
        image_ids = list(image_ids)
        sample_count = int(len(image_ids) * sample_ratio)
        sampled = random.sample(image_ids, min(sample_count, len(image_ids)))
        selected_image_ids.update(sampled)

    # 筛选图像和标注
    selected_images = [images[iid] for iid in selected_image_ids]
    selected_annotations = [
        ann
        for iid in selected_image_ids
        for ann in image_id_to_annotations[iid]
    ]

    new_coco = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories
    }

    # 写入新文件
    with open(output_file_path, 'w') as f:
        json.dump(new_coco, f, indent=2)

    print(f"[完成] 所有类别已按 {sample_ratio:.2f} 比例采样")
    print(f"图像数量：{len(selected_images)}，标注数量：{len(selected_annotations)}")
    print(f"输出文件：{output_file_path}")


# ✅ 示例调用
if __name__ == "__main__":
    sample_coco_uniform_ratio(
        coco_file_path="./data/ArTaxOr/annotations/test.json",
        sample_ratio=0.9
    )
