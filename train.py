import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from os import makedirs
import time
from tqdm import tqdm
from dataset import Scene_Dataset
from VggNet import VggNet
from data_augmentation import SSDAugmentation, SSDBaseTransform
from torchvision.models import vgg16

lables2name = {0:"建筑", 1:"森林", 2:"冰川", 3:"高山", 4:"大海", 5:"街景"}

# 加载预训练模型
def load_pretrained(path=None):
    try:
        model = VggNet()
        model.load_state_dict(torch.load(path))
    except:
        print("load pretrained model failed")
    return model

def train(epochs, data_dir, batch_size):
    ssdtransform = SSDBaseTransform()
    print("loading data for training...")
    train_set = Scene_Dataset(data_dir, mode="train", transform=ssdtransform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_set = Scene_Dataset(data_dir, mode="val", transform=ssdtransform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    for epoch in (range(epochs)):
        average_loss = []
        print(f"EPOCH: {epoch}/{epochs}")
        model.train()
        pbar = tqdm(train_loader)
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            average_loss.append(loss.item())
            optimizer.step()
            if batch_idx % 10 == 0:
                average = sum(average_loss)/len(average_loss)
                pbar.set_postfix({"train_loss": average})
                # print(outputs.shape, "output", outputs, "labels",labels)
        lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            total_samples, total_correct = 0, 0
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
            print(f"val_accuracy: {total_correct / total_samples}")
    torch.save(model.state_dict(), "output/model_epochs_20.pth")


def test(data_dir, batch_size):
    print("loading data for testing...")
    test_set = Scene_Dataset(data_dir, mode="test")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    model.eval()
    lables_list = []
    predicted_list = []
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            lables_list.append(labels)
            outputs = model(inputs)
            probap = outputs
            _, predicted = torch.max(outputs, 1)
            predicted_list.append(predicted)
        all_lables = torch.cat(lables_list)
        all_lables = all_lables.cpu().numpy()
        all_predicted = torch.cat(predicted_list)
        all_predicted = all_predicted.cpu().numpy()
        end_time = time.time()
        con_metrix = confusion_matrix(all_lables, all_predicted)
        print(f"accuracy_score:{accuracy_score(all_lables, all_predicted)}")
        print(f"f1_score:{f1_score(all_lables, all_predicted, average='micro')}")
        print(f"metrics:\n{con_metrix}")
        tp_sum = 0
        total_sum = 0
        for i in range(len(con_metrix[0])):
            tp_sum += con_metrix[i][i]
            total_sum += sum(con_metrix[i])
        print(f"tp_sum/total_sum: {tp_sum/total_sum}")
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
