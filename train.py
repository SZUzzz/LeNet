import copy
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from model import LeNet  # 假定你已经定义了 LeNet 模型


# 加载数据集
def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    train_data, val_data = Data.random_split(
        train_data,
        [round(0.8 * len(train_data)), round(0.2 * len(train_data))]
    )

    train_dataloader = Data.DataLoader(
        dataset=train_data,
        batch_size=128,
        shuffle=True,
        num_workers=0  # 改为 0，确保 Windows/Linux 都可用
    )

    val_dataloader = Data.DataLoader(
        dataset=val_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    return train_dataloader, val_dataloader


# 模型训练
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    since = time.time()

    early_stopping_counter = 0
    patience = 5  # 提前停止的容忍度

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # 初始化训练与验证过程参数
        train_loss, train_corrects, train_num = 0.0, 0, 0
        val_loss, val_corrects, val_num = 0.0, 0, 0

        # Training phase
        model.train()
        for b_x, b_y in train_dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device)

            # 前向传播
            output = model(b_x)
            loss = criterion(output, b_y)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练数据
            train_loss += loss.item() * b_x.size(0)
            preds = torch.argmax(output, dim=1)
            train_corrects += torch.sum(preds == b_y.data)
            train_num += b_x.size(0)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for b_x, b_y in val_dataloader:
                b_x, b_y = b_x.to(device), b_y.to(device)

                output = model(b_x)
                loss = criterion(output, b_y)

                # 统计验证数据
                val_loss += loss.item() * b_x.size(0)
                preds = torch.argmax(output, dim=1)
                val_corrects += torch.sum(preds == b_y.data)
                val_num += b_x.size(0)

        # 计算损失和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f"Train Loss: {train_loss_all[-1]:.4f}, Train Acc: {train_acc_all[-1]:.4f}")
        print(f"Val Loss: {val_loss_all[-1]:.4f}, Val Acc: {val_acc_all[-1]:.4f}")

        # 保存最佳模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            early_stopping_counter = 0  # 重置早停计数
        else:
            early_stopping_counter += 1

        # 提前停止
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

    # 计算训练耗时
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")

    # 保存最佳模型
    os.makedirs('LeNet/model', exist_ok=True)
    torch.save(best_model_wts, 'LeNet/model/best_model.pth')
    model.load_state_dict(best_model_wts)

    # 保存训练过程
    train_process = pd.DataFrame({
        "epoch": range(1, len(train_loss_all) + 1),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    })

    return train_process


# 绘图函数
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))

    # 绘制 Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], "ro-", label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], "bs-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    # 绘制 Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], "ro-", label="Train Acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], "bs-", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.show()


if __name__ == '__main__':
    # 实例化模型
    model = LeNet()

    # 加载数据
    train_dataloader, val_dataloader = train_val_data_process()

    # 训练模型
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=60)

    # 绘制结果
    matplot_acc_loss(train_process)
