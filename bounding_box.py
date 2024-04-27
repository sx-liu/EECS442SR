import cv2
import numpy as np

# 确保坐标在图像尺寸范围内
def get_location(val, length):
    if val < 0:
        return 0
    elif val > length:
        return length
    else:
        return val

# 读取图像
image_path = '/nfs/turbo/coe-chaijy/xuejunzh/CodeFormer/outputs/final_results/00.png'  # 使用上传图像的路径
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# 图像的原始高度和宽度
original_h, original_w = image.shape[:2]

# 检测到的面部数据（针对 512x811 尺寸的图像）
det_faces = [
    np.array([470.7142, 140.22777, 522.8775, 208.20227], dtype=np.float32),
    np.array([292.55606, 130.32878, 348.89777, 207.18552], dtype=np.float32),
    np.array([92.83741, 137.1806, 164.81105, 214.59409], dtype=np.float32),
    np.array([648.6708, 114.33363, 711.4867, 196.5089], dtype=np.float32)
]

# 计算缩放因子
scale_w = original_w / 811
scale_h = original_h / 512

# 应用缩放因子到检测到的面部坐标
scaled_faces = []
for face in det_faces:
    scaled_face = np.array(face, dtype=np.float32)
    scaled_face[[0, 2]] *= scale_w  # 缩放宽度坐标
    scaled_face[[1, 3]] *= scale_h  # 缩放高度坐标
    scaled_faces.append(scaled_face)

# 在图像上标示面部区域
for det_face in scaled_faces:
    left = get_location(int(det_face[0]), original_w)
    top = get_location(int(det_face[1]), original_h)
    right = get_location(int(det_face[2]), original_w)
    bottom = get_location(int(det_face[3]), original_h)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # 红色矩形，2px宽

# 将带有边界框的图像保存到文件
output_image_path = 'faces_visualized.jpg'
cv2.imwrite(output_image_path, image)
print(f"Image with visualized face bounding boxes saved to {output_image_path}")
