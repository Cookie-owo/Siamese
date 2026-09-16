import os
import argparse
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import json
import platform
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data.dataset import POCTDataset
from models.attention_resnet34 import AttentionResNet34
from reproducibility.reproducibility import create_generator, seed_worker, set_global_seed
from config import load_config
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


def get_argparse():
    parser = argparse.ArgumentParser(
        description=(
            "Train the two-attention ResNet-34 model "
            "using paired RGB and attention-map inputs."
        )
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Dataset root containing train/UT, train/T, "
            "val/UT and val/T directories."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/attention_resnet34_layer3_layer4",
        help="Directory used to save weights, logs and training figures.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=2.0,
        help="Positive-class weight used by BCEWithLogitsLoss.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=3.0,
        help="K value in the attention threshold: mean - K * standard deviation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, PyTorch and DataLoader workers.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic cuDNN execution; training may be slower.",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable online training-data augmentation.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration to validate and archive.",
    )

    return parser


def write_run_provenance(args):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.config:
        config_path = os.path.abspath(args.config)
        load_config(config_path)
        shutil.copyfile(config_path, os.path.join(args.save_dir, "config_used.yaml"))
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "positive_class_weight": args.pos_weight,
        "reference_map_K": args.k,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "augmentation_enabled": not args.no_augment,
    }
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(args.save_dir, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(environment, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with open(os.path.join(args.save_dir, "command.txt"), "w", encoding="utf-8") as handle:
        handle.write(" ".join(sys.argv) + "\n")


def apply_config_defaults(args):
    """Apply YAML training values while preserving explicit CLI overrides."""
    if not args.config:
        return args
    config = load_config(args.config)
    training = config.get("training", {})
    optimizer = training.get("optimizer", {})
    loss = training.get("loss", {})
    values = {
        "epochs": training.get("epochs"),
        "batch_size": training.get("batch_size"),
        "lr": optimizer.get("initial_learning_rate"),
        "pos_weight": loss.get("positive_class_weight"),
        "k": config.get("preprocessing", {}).get("attention_map", {}).get("K"),
        "seed": training.get("random_seed"),
    }
    parser_defaults = {
        "epochs": 100,
        "batch_size": 32,
        "lr": 1e-4,
        "pos_weight": 2.0,
        "k": 3.0,
        "seed": 42,
    }
    for name, value in values.items():
        if value is not None and getattr(args, name) == parser_defaults[name]:
            setattr(args, name, value)
    if training.get("deterministic_execution") and not args.deterministic:
        args.deterministic = True
    if training.get("online_augmentation") is False and not args.no_augment:
        args.no_augment = True
    return args


def visualize_augmented_samples(dataset, save_path, num_samples=5):
    """可视化数据增强后的样本，显示原始图像和增强后的图像"""
    try:
        # 随机选择样本进行可视化
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        # 创建子图：每行显示一个样本的原始图像和增强后的图像
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(indices):
            try:
                # 获取数据
                (img_tensor, att_tensor), label = dataset[idx]

                # 转换tensor为numpy数组用于显示
                img_np = img_tensor.permute(1, 2, 0).numpy()

                # 反归一化（假设使用了ImageNet标准化）
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)

                # 显示增强后的图像
                axes[i, 0].imshow(img_np)
                axes[i, 0].set_title(f'Augmented Sample {i + 1}\nLabel: {"T" if label == 1 else "UT"}', fontsize=10)
                axes[i, 0].axis('off')

                # 显示attention tensor（如果有的话）
                if att_tensor is not None:
                    att_np = att_tensor.squeeze().numpy()
                    axes[i, 1].imshow(att_np, cmap='hot')
                    axes[i, 1].set_title(f'Attention Map {i + 1}', fontsize=10)
                else:
                    axes[i, 1].text(0.5, 0.5, 'No Attention Map', ha='center', va='center',
                                    transform=axes[i, 1].transAxes)
                    axes[i, 1].set_title(f'No Attention {i + 1}', fontsize=10)
                axes[i, 1].axis('off')

            except Exception as e:
                axes[i, 0].text(0.5, 0.5, f'Error loading\n{str(e)[:50]}...', ha='center', va='center',
                                transform=axes[i, 0].transAxes)
                axes[i, 0].set_title(f'Error {i + 1}', fontsize=10)
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[save] Augmentation visualization: {save_path}")

    except Exception as e:
        print(f"可视化过程中出现错误: {str(e)}")
        # 创建一个简单的错误报告图
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Data Augmentation Visualization Failed\n\nError: {str(e)}',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Visualization Error Report')
        ax.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[save] Error report saved: {save_path}")


def save_training_curves(train_loss, train_acc, val_acc, val_loss, lr_list, save_path):
    plt.figure(figsize=(18, 12))

    # 1. 损失
    plt.subplot(2, 3, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 准确率
    plt.subplot(2, 3, 2)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    # 3. 学习率
    plt.subplot(2, 3, 3)
    plt.plot(lr_list, label='LR')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.title('Learning Rate Schedule')
    plt.grid(True)

    # 4. 损失-准确率联合视图
    plt.subplot(2, 3, 4)
    plt.scatter(train_loss, train_acc, c=range(len(train_loss)), cmap='viridis', label='Train')
    plt.scatter(val_loss, val_acc, c=range(len(val_loss)), cmap='viridis', marker='x', label='Val')
    plt.colorbar(label='Epoch')
    plt.xlabel('Loss')
    plt.ylabel('Accuracy')
    plt.title('Loss-Accuracy Correlation')
    plt.legend()
    plt.grid(True)

    # 5. 移动平均准确率
    win = max(1, len(train_acc) // 10)
    ma_train = np.convolve(train_acc, np.ones(win) / win, mode='valid')
    ma_val = np.convolve(val_acc, np.ones(win) / win, mode='valid')
    plt.subplot(2, 3, 5)
    plt.plot(ma_train, label=f'Train MA({win})')
    plt.plot(ma_val, label=f'Val MA({win})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Moving Average Accuracy')
    plt.legend()
    plt.grid(True)

    # 6. 最佳 epoch
    best_epoch = int(np.argmax(val_acc))
    plt.subplot(2, 3, 6)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Val')
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Best Model @ {best_epoch}, Val Acc {val_acc[best_epoch]:.4f}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_file = os.path.join(save_path, 'enhanced_training_curves.png')
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"[save] {save_file}")
    
def export_curves_vector_and_history(train_loss, train_acc, val_loss, val_acc, lr_list, save_dir):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # ---------- 保存历史数据 ----------
    np.savez(os.path.join(save_dir, 'history_arrays.npz'),
             train_loss=np.array(train_loss, dtype=float),
             train_acc=np.array(train_acc, dtype=float),
             val_loss=np.array(val_loss, dtype=float),
             val_acc=np.array(val_acc, dtype=float),
             lr=np.array(lr_list, dtype=float))

    # 也顺手存成 CSV（方便用别的工具画）
    np.savetxt(os.path.join(save_dir, 'train_acc.csv'), np.array(train_acc, dtype=float), delimiter=',')
    np.savetxt(os.path.join(save_dir, 'val_acc.csv'),   np.array(val_acc,   dtype=float), delimiter=',')
    np.savetxt(os.path.join(save_dir, 'train_loss.csv'),np.array(train_loss,dtype=float), delimiter=',')
    np.savetxt(os.path.join(save_dir, 'val_loss.csv'),  np.array(val_loss,  dtype=float), delimiter=',')

    # ---------- 矢量化导出：单张 Accuracy 图 ----------
    best_epoch = int(np.argmax(val_acc)) if len(val_acc) > 0 else -1

    plt.figure(figsize=(5, 5))
    if len(train_acc) > 0: plt.plot(train_acc, label='Train')
    if len(val_acc)   > 0: plt.plot(val_acc,   label='Val')
    if best_epoch >= 0:
        plt.axvline(best_epoch, linestyle='--', label=f'Best Epoch {best_epoch}')
        plt.title(f'Best Model @ {best_epoch}, Val Acc {val_acc[best_epoch]:.4f}')
    else:
        plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_curve.svg'), format='svg')  # 矢量
    plt.savefig(os.path.join(save_dir, 'acc_curve.pdf'))                # 矢量
    plt.savefig(os.path.join(save_dir, 'acc_curve.png'), dpi=300)       # 位图
    plt.close()




def train(args):
    args = apply_config_defaults(args)
    args.data_root = os.path.abspath(args.data_root)
    args.save_dir = os.path.abspath(args.save_dir)
    write_run_provenance(args)

    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than zero.")

    if args.lr <= 0:
        raise ValueError("--lr must be greater than zero.")

    if args.k < 0:
        raise ValueError("--k must be non-negative.")

    required_directories = [
        os.path.join(args.data_root, split, class_name)
        for split in ("train", "val")
        for class_name in ("UT", "T")
    ]

    missing_directories = [
        directory
        for directory in required_directories
        if not os.path.isdir(directory)
    ]

    if missing_directories:
        missing_text = "\n".join(
            f"  - {directory}"
            for directory in missing_directories
        )
        raise FileNotFoundError(
            "Required dataset directories are missing:\n"
            f"{missing_text}"
        )

    set_global_seed(
        args.seed,
        deterministic=args.deterministic,
    )
    os.makedirs(os.path.join(args.save_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'training'), exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集 - 根据参数决定是否使用增强
    train_mode = 'train' if not args.no_augment else 'val'  # 'train'模式使用增强，'val'模式不使用
    train_set = POCTDataset(
        root_dir=os.path.join(args.data_root, 'train'),
        mode=train_mode,  # 控制是否使用增强
        K=args.k
    )
    val_set = POCTDataset(
        root_dir=os.path.join(args.data_root, 'val'),
        mode='val',  # 验证集永远不使用增强
        K=args.k
    )

    # 简洁的数据增强信息显示
    print(f"\n=== 训练配置信息 ===")
    print(f"数据集名称: {args.data_root}")
    print("模型类型: AttentionResNet34 (attention at layer3 and layer4)")
    print(f"训练样本数: {len(train_set)}")
    print(f"验证样本数: {len(val_set)}")
    print(f"数据增强: {'启用' if not args.no_augment else '禁用'}")
    print(f"Attention threshold K: {args.k}")
    print(f"Random seed: {args.seed}")
    print(f"Deterministic mode: {args.deterministic}")
    if not args.no_augment:
        print(f"增强策略: 在线随机增强")
        print(f"包含: 旋转(80%) + 对比度调整(50%) + 高斯噪声(50%)")
    print("==================\n")

    # 数据增强可视化
    if not args.no_augment:
        print("正在生成数据增强可视化...")
        try:
            # 创建可视化保存目录
            viz_save_path = os.path.join(args.save_dir, 'training', 'data_augmentation_samples.png')
            os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)
            
            # 调用可视化函数
            visualize_augmented_samples(train_set, viz_save_path, num_samples=5)
            print(f"数据增强可视化已保存到: {viz_save_path}")
        except Exception as e:
            print(f"数据增强可视化失败: {str(e)}")
            print("继续训练...")
    else:
        print("数据增强已禁用，跳过可视化")

    # Reset random states because augmentation visualization above
    # consumes Python, NumPy and PyTorch random numbers.
    set_global_seed(
        args.seed,
        deterministic=args.deterministic,
    )

    num_workers = min(
        os.cpu_count() or 1,
        args.batch_size if args.batch_size > 1 else 0,
        8,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=create_generator(args.seed),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=create_generator(args.seed + 1),
    )

    # 模型
    model = AttentionResNet34(num_class=1).to(device)

    # 损失、优化器、调度器
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

    # 记录列表
    train_loss_list, train_acc_list, val_acc_list, val_loss_list, lr_list = [], [], [], [], []

    best_val_acc = float("-inf")
    total_time = 0.0

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # ---------- 训练 ----------
        model.train()
        epoch_train_loss = 0.0
        correct_train = total_train = 0
        train_bar = tqdm(train_loader, desc=f'Train Epoch {epoch + 1}/{args.epochs}')
        for (img, att), labels in train_bar:
            img = img.to(device)
            att = att.to(device)
            labels = labels.float().to(device).reshape(-1)

            logits = model(img, att).reshape(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            train_bar.set_postfix({'loss': f"{loss.item():.4f}",
                                   'acc': f"{correct_train / total_train:.3f}"})

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # ---------- 验证 ----------
        model.eval()
        correct_val = total_val = 0
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Val Epoch {epoch + 1}/{args.epochs}')
        with torch.no_grad():
            for (img, att), labels in val_bar:
                img = img.to(device)
                att = att.to(device)
                labels = labels.float().to(device).reshape(-1)

                logits = model(img, att).reshape(-1)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                val_bar.set_postfix({'acc': f"{correct_val / total_val:.3f}"})

        val_acc = correct_val / total_val
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        lr_list.append(optimizer.param_groups[0]['lr'])
        scheduler.step(val_acc)

        # 保存权重
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'weights', 'last_epoch.pth'))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'weights', 'best_model.pth'))
            print(f"New best model saved with acc: {best_val_acc:.4f}")

        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s")

    # ---------- 训练结束 ----------
    print(f"\nTraining completed in "
          f"{total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    # Reload the best validation checkpoint before final evaluation.
    best_model_path = os.path.join(
        args.save_dir,
        "weights",
        "best_model.pth",
    )

    if not os.path.isfile(best_model_path):
        raise FileNotFoundError(
            "Best-model checkpoint was not created: "
            f"{best_model_path}"
        )

    best_state_dict = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        best_state_dict,
        strict=True,
    )
    model.eval()

    print(f"Reloaded best model: {best_model_path}")

    # 生成增强曲线
    save_training_curves(train_loss_list, train_acc_list,
                         val_acc_list, val_loss_list, lr_list,
                         os.path.join(args.save_dir, 'training'))
                         
    export_curves_vector_and_history(train_loss_list, train_acc_list,
                                     val_loss_list, val_acc_list, lr_list,
                                     os.path.join(args.save_dir, 'training'))

    # ---------- 日志 ----------
    log_path = os.path.join(args.save_dir, 'training', 'training_log.txt')
    with open(log_path, 'w') as f:
        f.write(f"Training Configuration:\n")
        f.write(f"Dataset Name: {args.data_root}\n")
        f.write("Model Type: AttentionResNet34 (attention at layer3 and layer4)\n")
        f.write(f"Epochs: {args.epochs}\nBatch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Positive Weight: {args.pos_weight}\n")
        f.write(f"Attention Threshold K: {args.k}\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Deterministic Mode: {args.deterministic}\n")
        f.write(f"Data Augmentation: {'Enabled' if not args.no_augment else 'Disabled'}\n")
        f.write(f"Augmentation Type: {'Online Random Augmentation' if not args.no_augment else 'None'}\n")
        f.write(f"Train Samples: {len(train_set)}\n")
        f.write(f"Validation Samples: {len(val_set)}\n")
        f.write(f"Best Val Acc: {best_val_acc:.4f}\n\n")

    # ---------- 随机 5 个验证样本 ----------
    model.eval()
    sample_lines = ["=== Sample Predictions ===\n"]
    with torch.no_grad():
        for (img, att), labels in val_loader:
            img = img.to(device)
            att = att.to(device)
            labels = labels.to(device).reshape(-1)

            logits = model(img, att).reshape(-1)
            probs = torch.sigmoid(logits)

            for i in range(min(5, labels.size(0))):
                sample_lines.append(
                    f"Sample {i + 1}:\n"
                    f"  True Label: {'Positive' if labels[i].item() else 'Negative'}\n"
                    f"  Predicted Prob: {probs[i].item():.4f}\n"
                    f"  Prediction: {'Positive' if probs[i] > 0.5 else 'Negative'}\n\n"
                )
            break
    with open(log_path, 'a') as f:
        f.writelines(sample_lines)

    # ---------- 最终混淆矩阵 ----------
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for (img, att), labels in val_loader:
            img = img.to(device)
            att = att.to(device)
            labels = labels.to(device).reshape(-1)

            logits = model(img, att).reshape(-1)
            preds = (torch.sigmoid(logits) > 0.5).to(torch.int64)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(
                labels.to(torch.int64).cpu().tolist()
            )

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=[0, 1],
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Final Validation Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_path = os.path.join(args.save_dir, 'training', 'final_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[save] {cm_path}")


if __name__ == '__main__':
    train(get_argparse().parse_args())
