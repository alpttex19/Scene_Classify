import torch
from torch.utils.data import DataLoader
from os import makedirs
import time
from tqdm import tqdm
from data import Scene_Dataset
from VggNet import VggNet
from torchvision.models import vgg16

# 加载预训练模型
def load_pretrained(path=None):
    try:
        model = VggNet()
        model.load_state_dict(torch.load(path))
    except:
        print("load pretrained model failed")
    return model

def train(epochs, data_dir, batch_size):
    print("loading data for training...")
    train_set = Scene_Dataset(data_dir, mode="train")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_set = Scene_Dataset(data_dir, mode="val")
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    average_loss = []
    for epoch in range(epochs):
        print(f"EPOCH: {epoch}/{epochs}")
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            average_loss.append(loss.item())
            optimizer.step()
            if batch_idx % 50 == 0:
                average = sum(average_loss)/len(average_loss)
                print(f"batch_idx: {batch_idx}, train_loss: {average}")
                average_loss.clear()
                # print(outputs.shape, "output", outputs, "labels",labels)
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            total_samples, total_correct = 0, 0
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            print(f"val_accuracy: {total_correct / total_samples}")
    torch.save(model.state_dict(), "output/model.pth")


def test(data_dir, batch_size):
    print("loading data for testing...")
    test_set = Scene_Dataset(data_dir, mode="test")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    model.eval()
    with torch.no_grad():
        total_samples, total_correct = 0, 0
        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        print(f"test_accuracy: {total_correct / total_samples}")
        end_time = time.time()
        print(f"test time: {end_time-start_time}")
        print(f"total samples: {test_set.__len__()}")


if __name__ == "__main__":
    # 命令行读取参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/xinhuang/Desktop/arapat/Scene_Classify/dataset")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pretrained_model", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.pretrained_model is None:
        model = VggNet().to(device)
    else:
        model = load_pretrained(args.pretrained_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    makedirs(args.output_dir, exist_ok=True)
    if args.mode == "train":
        train(args.epochs, args.data_dir, args.batch_size)
    else:
        test(args.data_dir, args.batch_size)
