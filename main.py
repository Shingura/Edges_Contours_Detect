import os
import cv2
import numpy as np

OUTPUT_DIR = "output"

def detect_edges_and_contours(img):
    # 灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊降噪（卷积核大小为 3 x 3）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny 对降噪后的灰度图生成二值边缘图
    edges = cv2.Canny(blurred, 50, 150)

    # 把灰度图转换成黑白图，白色前景、纯黑背景
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    # 绕行白色图边界，得到边缘图
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 灰度图 + 边缘（单通道灰度图无法上色，需要转回三通道）
    gray_with_edges = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    gray_with_edges[edges > 0] = (0, 255, 0)

    # 黑底纯轮廓
    outline = np.zeros_like(img)
    cv2.drawContours(outline, contours, -1, (0, 255, 0), 1)

    # 黑底纯边缘
    edges_only = np.zeros_like(img)
    edges_only[edges > 0] = (0, 255, 0)

    return gray, gray_with_edges, outline, edges_only

maimai = cv2.imread("maimaidx_circle_plus.png")
ow2 = cv2.imread("ow2.png")

os.makedirs(OUTPUT_DIR, exist_ok = True)

results = {
    "maimai": detect_edges_and_contours(maimai),
    "ow2": detect_edges_and_contours(ow2),
}

for name, (gray_img, gray_with_edges, outline, edges_only) in results.items():
    cv2.imshow(f"{name} - 灰度图", gray_img)
    cv2.imshow(f"{name} - 灰度图+边缘", gray_with_edges)
    cv2.imshow(f"{name} - 纯轮廓", outline)
    cv2.imshow(f"{name} - 纯边缘", edges_only)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_gray.png"), gray_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_gray_edges.png"), gray_with_edges)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_contours.png"), outline)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_edges_only.png"), edges_only)

cv2.waitKey(0)
cv2.destroyAllWindows()
