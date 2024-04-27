import cv2
import numpy as np

def get_location(val, length):
    """Ensure the coordinate stays within the image dimensions."""
    if val < 0:
        return 0
    elif val > length:
        return length
    else:
        return val

def apply_gaussian_blur(image, kernel_size=(3, 3)):
    """Apply Gaussian blur to the image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_sharpening_filter(image, kernel):
    """Apply the sharpening filter to the image."""
    return cv2.filter2D(image, -1, kernel)

def mask_faces_and_sharpen(image, scaled_faces, w, h):
    """Sharpen the image while protecting the detected faces."""
    # 创建一个与图像相同大小的遮罩，所有区域默认不显示（黑色）
    mask = np.zeros(image.shape[:2], dtype="uint8")

    # 对于每个脸部区域，我们将其设置为白色（255），表示这部分保持不变
    for det_face in scaled_faces:
        left = get_location(int(det_face[0]), w)
        right = get_location(int(det_face[2]), w)
        top = get_location(int(det_face[1]), h)
        bottom = get_location(int(det_face[3]), h)
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

    # 应用锐化滤镜到非面部区域
    blurred_image = apply_gaussian_blur(image)
    # 然后对非面部区域应用锐化
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 温和的锐化核
    sharpened_image = apply_sharpening_filter(blurred_image, sharpening_kernel)
    # 结合锐化图像和原图像，面部区域保持原样
    final_image = cv2.bitwise_and(sharpened_image, sharpened_image, mask=255-mask)
    final_image += cv2.bitwise_and(image, image, mask=mask)

    return final_image


image = cv2.imread('/nfs/turbo/coe-chaijy/xuejunzh/CodeFormer/outputs/final_results/00.png')
if image is None:
    print(f"Error: Unable to load image at {image}")
    exit()

h, w = image.shape[:2]

# 假设 det_faces 已由某种人脸检测模型给出，如下所示：
det_faces = [
    np.array([470.7142, 140.22777, 522.8775, 208.20227, 0.79962605], dtype=np.float32),
    np.array([292.55606, 130.32878, 348.89777, 207.18552, 0.79893553], dtype=np.float32),
    np.array([92.83741, 137.1806, 164.81105, 214.59409, 0.79786694], dtype=np.float32),
    np.array([648.6708, 114.33363, 711.4867, 196.5089, 0.7965852], dtype=np.float32)
]
# 计算缩放因子
scale_w = w / 811
scale_h = h / 512

# 应用缩放因子到检测到的面部坐标
scaled_faces = []
for face in det_faces:
    scaled_face = face.copy()
    scaled_face[0] *= scale_w  # 缩放宽度坐标
    scaled_face[1] *= scale_h  # 缩放高度坐标
    scaled_face[2] *= scale_w
    scaled_face[3] *= scale_h
    scaled_faces.append(scaled_face)

# 处理图像
final_image = mask_faces_and_sharpen(image, scaled_faces, w, h)

# 保存结果图像到文件
output_path = 'sharpened_protected_image.jpg'  # Update to the desired output path
cv2.imwrite(output_path, final_image)
