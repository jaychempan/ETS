import json
import os
import random
import cv2
from tqdm import tqdm

def convert_and_annotate(coco_file_path, image_directory, score_threshold=0.0):
    # 读取 COCO JSON 文件
    with open(coco_file_path, 'r') as file:
        coco_data = json.load(file)

    # 生成文件名和图像信息的映射
    images_info = {image['id']: image for image in coco_data['images']}

    # 生成类别ID到类别名称的映射
    category_id_to_name = {category['id']: category['name'] for category in coco_data['categories']}

    # 生成 JSONL 数据
    jsonl_data = []

    for image_id, image_info in tqdm(images_info.items(), desc=f"Processing annotations from {os.path.basename(coco_file_path)}"):
        instances = []
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                # 如果包含 score 且小于阈值，则跳过
                if 'score' in annotation and annotation['score'] < score_threshold:
                    continue
                x1, y1, w, h = annotation['bbox']
                bbox_xyxy = [x1, y1, x1 + w, y1 + h]
                instance = {
                    "bbox": bbox_xyxy,
                    "category": category_id_to_name[annotation['category_id']]
                }
                if 'score' in annotation:
                    instance["score"] = annotation['score']
                instances.append(instance)
        
        jsonl_data.append({
            "filename": image_info['file_name'],
            "height": image_info['height'],
            "width": image_info['width'],
            "detection": {"instances": instances}
        })

    # 自动生成 JSONL 文件路径
    jsonl_file_path = coco_file_path.replace('.json', '.jsonl')

    # 保存为 JSONL 文件
    with open(jsonl_file_path, 'w') as jsonl_file:
        for entry in jsonl_data:
            jsonl_file.write(json.dumps(entry) + '\n')
    
    # 获取 image_directory 下的所有图像文件（不包括子目录）
    def get_image_files(directory):
        image_files = []
        for file in os.listdir(directory):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                image_files.append(os.path.join(directory, file))
        return image_files

    # 获取 JSONL 文件中所有的图像文件名
    def get_image_filenames_from_jsonl(annotations_path):
        image_filenames = set()
        with open(annotations_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                image_filenames.add(data['filename'])
        return image_filenames

    # 绘制边框
    def plot_one_box(x, img, color=None, label=None, line_thickness=2):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2_label = c1[0] + t_size[0], c1[1] + t_size[1] + 5
            cv2.rectangle(img, c1, c2_label, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] + 15), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    # 绘制所有边框到图片
    def draw_boxes(image_path, annotations_path):
        image = cv2.imread(image_path)

        with open(annotations_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if data['filename'] == os.path.basename(image_path):
                    for instance in data['detection']['instances']:
                        score = instance.get('score', 1.0)
                        if score < score_threshold:
                            continue
                        bbox = instance['bbox']
                        label = f"{instance['category']} {score:.2f}" if 'score' in instance else instance['category']
                        plot_one_box(bbox, image, label=label)

        # 保存带标注的图片
        save_path = os.path.join(os.path.dirname(image_path), 'annotated_images')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file_name = os.path.splitext(os.path.basename(image_path))[0] + '_annotated.jpg'
        output_file_path = os.path.join(save_path, output_file_name)

        cv2.imwrite(output_file_path, image)
        print(f"Annotated image saved to {output_file_path}")

    # 处理所有图片
    def process_images_in_directory(directory, annotations_path):
        image_files = get_image_files(directory)
        jsonl_filenames = get_image_filenames_from_jsonl(annotations_path)

        for image_path in image_files:
            if os.path.basename(image_path) in jsonl_filenames:
                draw_boxes(image_path, annotations_path)

    # 处理指定目录下的所有图像文件
    process_images_in_directory(image_directory, jsonl_file_path)
    print(f"JSONL format data has been saved to {jsonl_file_path}")

# 使用示例
coco_file_path = '/home/panjiancheng/projects/NTIRE2025-CDFSOD-AI4EarthLab/data/ArTaxOr/annotations/test_val_0.3.json'
image_directory = '/home/panjiancheng/projects/NTIRE2025-CDFSOD-AI4EarthLab/data/ArTaxOr/test/'
score_threshold = 0.0  # 只显示置信度大于等于0.3的框

convert_and_annotate(coco_file_path, image_directory, score_threshold=score_threshold)
