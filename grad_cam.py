import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from VggNet import VggNet

# 加载预训练模型
def load_pretrained(path=None):
    try:
        model = VggNet()
        model.load_state_dict(torch.load(path))
    except:
        print("load pretrained model failed")
    return model

# 预处理图像
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path)
    img = preprocess(img).unsqueeze(0)
    return img

# 定义钩子函数来获取梯度和激活
gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, input, output):
    global activations
    activations = output


# 计算Grad-CAM热力图
def generate_cam(activations, gradients):
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap+1e-10)
    return heatmap

# 可视化Grad-CAM
def visualize_cam(img_path, cam, output_path):
    img = cv2.imread(img_path)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.8, heatmap, 0.2, 0)

    cv2.imwrite(output_path, superimposed_img)

# 可以放在训练循环中，用来可视化Grad-CAM
def visualize_process(model, epoch, batch_idx, device):
    model.eval()
    target_layer = model.Conv5.conv[-3]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    import pandas as pd
    df = pd.read_csv("./dataset/test_data.csv")
    img_name_list = df['image_name'].tolist()
    label_list = df['label'].tolist()
    for i in range(20):#len(label_list)):
        img_name = img_name_list[i]
        lable = label_list[i]
        img_path = f"./dataset/imgs/{img_name}"
        input_tensor = preprocess_image(img_path)
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        # print(f'Predicted: {pred.item()}')
        # 向后传播以获取梯度
        model.zero_grad()
        class_score = output[0, pred.item()]
        class_score.backward()
        # 生成并显示热力图
        cam = generate_cam(activations, gradients)
        output_path = f"./output/cam_map/{img_name[:-4]}_{epoch}_{batch_idx}.jpg"
        visualize_cam(img_path, cam, output_path)

# 测试Grad-CAM
if __name__== "__main__":
    lables2name = {0:"建筑", 1:"森林", 2:"冰川", 3:"高山", 4:"大海", 5:"街景"}
    model = load_pretrained('./output/model_epochs_20.pth')
    model.eval()
    target_layer = model.Conv5.conv[-3]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    image_root = f"./dataset/imgs"
    import pandas as pd
    df = pd.read_csv("./dataset/train_data.csv")
    img_name_list = df['image_name'].tolist()
    label_list = df['label'].tolist()
    for i in range(len(label_list)):
        if i > 20:
            break
        img_name = img_name_list[i]
        lable = label_list[i]
        print(f'processing {img_name}...')  
        img_path = f"./dataset/imgs/{img_name}"
        input_tensor = preprocess_image(img_path)
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        # print(f'Predicted: {pred.item()}')
        # 注册钩子在最后一个卷积层
        # 向后传播以获取梯度
        model.zero_grad()
        class_score = output[0, pred.item()]
        class_score.backward()
        # 生成并显示热力图
        cam = generate_cam(activations, gradients)
        output_path = f"./output/cam_map/{img_name[:-4]}_lab{lables2name[lable]}_pred{lables2name[pred.item()]}.jpg"
        visualize_cam(img_path, cam, output_path)