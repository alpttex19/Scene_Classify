import os
from PIL import Image

# 获取 "imgs" 目录下的所有文件
img_dir = "imgs"
imgs = os.listdir(img_dir)

# 创建 "Images" 目录
if not os.path.exists("Images"):
    os.makedirs("images")

# 遍历所有文件
for img_name in imgs:
    # 打开图片
    img = Image.open(os.path.join(img_dir, img_name))
    # 改变图片的大小
    img_resized = img.resize((224, 224))
    # 保存图片到 "Images" 目录下
    img_resized.save(os.path.join("Images", img_name))