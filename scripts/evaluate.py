import os
import argparse
import csv
import pathlib
import sys
from pathlib import Path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import hashlib
import json
import platform
import shutil
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from data.dataset import POCTDataset
from models.attention_resnet34 import AttentionResNet34
from config import load_config
import matplotlib
matplotlib.use('Agg')



def get_args():
    parser = argparse.ArgumentParser(description='POCT Network Testing')
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Test-set directory containing UT and T subdirectories.",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to the trained AttentionResNet34 state dictionary.",
    )
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold')
    parser.add_argument(
        "--k",
        type=float,
        default=3.0,
        help=(
            "K value used to generate the attention map: "
            "mean - K * standard deviation."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/attention_test",
        help="Directory used to save test metrics and figures.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML configuration to validate and archive.",
    )
    return parser.parse_args()


def write_run_provenance(args):
    if args.config:
        config_path = os.path.abspath(args.config)
        load_config(config_path)
        shutil.copyfile(config_path, os.path.join(args.output_dir, "config_used.yaml"))
    digest = hashlib.sha256()
    with open(args.weights_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with open(os.path.join(args.output_dir, "checkpoint_sha256.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"{digest.hexdigest()}  {os.path.basename(args.weights_path)}\n")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "threshold": args.threshold,
        "reference_map_K": args.k,
        "batch_size": args.batch_size,
    }
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)
    with open(os.path.join(args.output_dir, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(environment, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def setup_directories(output_dir):
    """创建测试结果文件夹"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    return {
        'main_dir': output_dir,
        'vis_dir': os.path.join(output_dir, 'figures')
    }



def load_model(weights_path, device):
    """Load a checkpoint only when every parameter matches."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights file does not exist: {weights_path}"
        )

    try:
        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError(
            "Checkpoint is not a non-empty state dictionary."
        )

    # Support both a plain state_dict and common checkpoint containers.
    if all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        state_dict = checkpoint
    elif isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(
        checkpoint.get("model_state_dict"),
        dict,
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        raise TypeError(
            "Checkpoint does not contain a valid state dictionary."
        )

    # Remove the prefix produced by torch.nn.DataParallel.
    clean_state_dict = {}

    for key, value in state_dict.items():
        clean_key = (
            key[7:]
            if key.startswith("module.")
            else key
        )

        if clean_key in clean_state_dict:
            raise KeyError(
                "Duplicate checkpoint key after normalization: "
                f"{clean_key}"
            )

        clean_state_dict[clean_key] = value

    model = AttentionResNet34(num_class=1)

    try:
        model.load_state_dict(
            clean_state_dict,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not exactly match "
            "models.attention_resnet34.AttentionResNet34."
        ) from exc

    model.to(device)
    model.eval()

    print(
        "[PASS] Strict checkpoint loading configured: "
        f"{len(clean_state_dict)} tensors"
    )

    return model




def evaluate_model(
    model,
    test_loader,
    device,
    threshold=0.5,
):
    """Evaluate the model without collapsing batch-size-one tensors."""
    all_probs = []
    all_preds = []
    all_labels = []
    all_filenames = [Path(sample[0]).name for sample in test_loader.dataset.samples]
    incorrect_samples = []

    model.eval()

    with torch.inference_mode():
        for (images, attention), labels in tqdm(
            test_loader,
            desc="Testing",
        ):
            images = images.to(
                device,
                non_blocking=True,
            )
            attention = attention.to(
                device,
                non_blocking=True,
            )

            # Always keep labels as a one-dimensional batch tensor.
            labels = labels.to(
                device,
                dtype=torch.float32,
                non_blocking=True,
            ).reshape(-1)

            # reshape(-1) preserves a one-dimensional tensor even
            # when the current batch contains only one sample.
            logits = model(
                images,
                attention,
            ).reshape(-1)

            probs = torch.sigmoid(logits)
            preds = (
                probs > threshold
            ).to(torch.int64)

            labels_int = labels.to(torch.int64)

            all_probs.extend(
                probs.detach().cpu().tolist()
            )
            all_preds.extend(
                preds.detach().cpu().tolist()
            )
            all_labels.extend(
                labels_int.detach().cpu().tolist()
            )

            incorrect_indices = torch.nonzero(
                preds != labels_int,
                as_tuple=False,
            ).reshape(-1)

            for index in incorrect_indices.tolist():
                incorrect_samples.append(
                    {
                        "image": images[index].detach().cpu(),
                        "attention": (
                            attention[index]
                            .detach()
                            .cpu()
                        ),
                        "prob": float(
                            probs[index].item()
                        ),
                        "pred": int(
                            preds[index].item()
                        ),
                        "label": int(
                            labels_int[index].item()
                        ),
                    }
                )

    if not all_labels:
        raise RuntimeError(
            "The test DataLoader produced no samples."
        )

    return {
        "probs": np.asarray(
            all_probs,
            dtype=np.float64,
        ),
        "preds": np.asarray(
            all_preds,
            dtype=np.int64,
        ),
        "labels": np.asarray(
            all_labels,
            dtype=np.int64,
        ),
        "filenames": all_filenames,
        "incorrect_samples": incorrect_samples,
    }




def save_results(
    results,
    dirs,
    threshold,
    args,
):
    """Save binary-classification metrics and visualizations."""
    labels = np.asarray(
        results["labels"],
        dtype=np.int64,
    ).reshape(-1)

    preds = np.asarray(
        results["preds"],
        dtype=np.int64,
    ).reshape(-1)

    probs = np.asarray(
        results["probs"],
        dtype=np.float64,
    ).reshape(-1)

    if labels.size == 0:
        raise ValueError(
            "Cannot save results because no test labels "
            "were produced."
        )

    if not (
        labels.size == preds.size == probs.size
    ):
        raise ValueError(
            "labels, preds, and probs must contain "
            "the same number of samples."
        )

    if not np.isin(labels, [0, 1]).all():
        raise ValueError(
            "Ground-truth labels must contain only 0 and 1."
        )

    if not np.isin(preds, [0, 1]).all():
        raise ValueError(
            "Predicted labels must contain only 0 and 1."
        )

    np.savez(
        os.path.join(
            dirs["main_dir"],
            "test_results.npz",
        ),
        probs=probs,
        preds=preds,
        labels=labels,
    )

    with open(os.path.join(dirs["main_dir"], "per_sample_predictions.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "filename", "true_label", "predicted_label", "predicted_probability", "correct"])
        for index, (filename, label, pred, prob) in enumerate(zip(results["filenames"], labels, preds, probs), 1):
            writer.writerow([index, filename, int(label), int(pred), f"{float(prob):.8f}", int(label == pred)])

    # Always produce a stable 2 x 2 binary confusion matrix,
    # including when one class is absent from the test subset.
    cm = confusion_matrix(
        labels,
        preds,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        int(value)
        for value in cm.ravel()
    )

    def safe_divide(
        numerator,
        denominator,
    ):
        if denominator == 0:
            return float("nan")

        return numerator / denominator

    def format_metric(value):
        if np.isnan(value):
            return "not available"

        return f"{value:.4f}"

    accuracy = safe_divide(
        tp + tn,
        tn + fp + fn + tp,
    )

    precision_positive = safe_divide(
        tp,
        tp + fp,
    )

    sensitivity = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    f1_positive = safe_divide(
        2 * tp,
        (2 * tp) + fp + fn,
    )

    # ROC-AUC and Average Precision require both classes for
    # a meaningful binary evaluation.
    has_both_classes = (
        np.unique(labels).size == 2
    )

    if has_both_classes:
        fpr, tpr, _ = roc_curve(
            labels,
            probs,
        )

        roc_auc = float(
            auc(
                fpr,
                tpr,
            )
        )

        precision_curve, recall_curve, _ = (
            precision_recall_curve(
                labels,
                probs,
            )
        )

        average_precision = float(
            average_precision_score(
                labels,
                probs,
            )
        )
    else:
        fpr = None
        tpr = None
        precision_curve = None
        recall_curve = None
        roc_auc = float("nan")
        average_precision = float("nan")

    report_path = os.path.join(
        dirs["main_dir"],
        "evaluation_report.txt",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Test Results "
            f"(Threshold={threshold:.4f})\n"
        )
        handle.write("=" * 50 + "\n")

        handle.write(
            f"Total samples: {labels.size}\n"
        )
        handle.write(
            "Negative samples (Actual): "
            f"{int((labels == 0).sum())}\n"
        )
        handle.write(
            "Positive samples (Actual): "
            f"{int((labels == 1).sum())}\n\n"
        )

        handle.write(
            "Accuracy: "
            f"{format_metric(accuracy)}\n"
        )
        handle.write(
            "Precision (Positive): "
            f"{format_metric(precision_positive)}\n"
        )
        handle.write(
            "Sensitivity / Recall (Positive): "
            f"{format_metric(sensitivity)}\n"
        )
        handle.write(
            "Specificity: "
            f"{format_metric(specificity)}\n"
        )
        handle.write(
            "F1-Score (Positive): "
            f"{format_metric(f1_positive)}\n"
        )
        handle.write(
            "ROC-AUC: "
            f"{format_metric(roc_auc)}\n"
        )
        handle.write(
            "Average Precision: "
            f"{format_metric(average_precision)}\n"
        )

        if not has_both_classes:
            handle.write(
                "Note: ROC-AUC and Average Precision "
                "are not available because the test labels "
                "contain only one class.\n"
            )

        handle.write(
            "\nConfusion Matrix Details:\n"
        )
        handle.write(
            f"True Negative: {tn}\n"
        )
        handle.write(
            f"False Positive: {fp}\n"
        )
        handle.write(
            f"False Negative: {fn}\n"
        )
        handle.write(
            f"True Positive: {tp}\n\n"
        )

        handle.write(
            "Classification Report:\n"
        )
        handle.write(
            classification_report(
                labels,
                preds,
                labels=[0, 1],
                target_names=[
                    "Negative",
                    "Positive",
                ],
                zero_division=0,
                digits=4,
            )
        )

    # 1. Confusion matrix
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Negative (0)",
            "Positive (1)",
        ],
        yticklabels=[
            "Negative (0)",
            "Positive (1)",
        ],
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            dirs["vis_dir"],
            "confusion_matrix.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # 2. ROC curve
    plt.figure(figsize=(8, 6))

    if has_both_classes:
        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                "ROC "
                f"(AUC={roc_auc:.4f})"
            ),
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
        )

        plt.legend(loc="lower right")
    else:
        plt.text(
            0.5,
            0.5,
            "ROC curve unavailable:\n"
            "only one class is present.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            dirs["vis_dir"],
            "roc_curve.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # 3. Precision-recall curve
    plt.figure(figsize=(8, 6))

    if has_both_classes:
        plt.plot(
            recall_curve,
            precision_curve,
            linewidth=2,
            label=(
                "PR "
                f"(AP={average_precision:.4f})"
            ),
        )

        plt.legend(loc="lower left")
    else:
        plt.text(
            0.5,
            0.5,
            "Precision-recall curve unavailable:\n"
            "only one class is present.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            dirs["vis_dir"],
            "precision_recall_curve.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # 4. 错误样本可视化（改进标签显示）
    if results['incorrect_samples']:
        num_samples = min(5, len(results['incorrect_samples']))
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        for i in range(num_samples):
            sample = results['incorrect_samples'][i]

            # 原始图像
            img = sample['image'].permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Label: {'Positive' if sample['label'] == 1 else 'Negative'}\n"
                                 f"Pred: {'Positive' if sample['pred'] == 1 else 'Negative'}")
            axes[0, i].axis('off')

            # 注意力图
            att = sample['attention'].squeeze().numpy()
            axes[1, i].imshow(att, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"Prob: {sample['prob']:.2f}")
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dirs['vis_dir'], 'incorrect_samples.png'))
        plt.close()

    # 5. 正确样本可视化（改进标签显示）
    correct_samples = [i for i, (pred, label) in enumerate(zip(results['preds'], results['labels']))
                       if pred == label]

    if correct_samples:
        np.random.shuffle(correct_samples)
        num_samples = min(5, len(correct_samples))
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

        test_set = POCTDataset(root_dir=args.data_root, mode='test', K=args.k)
        for plot_idx, sample_idx in enumerate(correct_samples[:num_samples]):
            sample = test_set[sample_idx]
            (img_tensor, att_tensor), label = sample

            # 原始图像
            img = img_tensor.permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            axes[0, plot_idx].imshow(img)
            axes[0, plot_idx].set_title(f"Label: {'Positive' if label == 1 else 'Negative'}\n"
                                        f"Pred: {'Positive' if results['preds'][sample_idx] == 1 else 'Negative'}")
            axes[0, plot_idx].axis('off')

            # 注意力图
            att = att_tensor.squeeze().numpy()
            axes[1, plot_idx].imshow(att, cmap='gray', vmin=0, vmax=1)
            axes[1, plot_idx].set_title(f"Prob: {results['probs'][sample_idx]:.2f}")
            axes[1, plot_idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(dirs['vis_dir'], 'correct_samples.png'))
        plt.close()



def test(args):
    """主测试函数"""
    dirs = setup_directories(args.output_dir)
    write_run_provenance(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.weights_path}...")
    model = load_model(args.weights_path, device)

    test_set = POCTDataset(root_dir=args.data_root, mode='test', K=args.k)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=min(os.cpu_count(), 4))

    # 添加数据统计
    negative_count = sum(1 for _, label in test_set.samples if label == 0)
    positive_count = sum(1 for _, label in test_set.samples if label == 1)
    print(f"Test samples: {len(test_set)}")
    print(f"Negative samples (UT): {negative_count}")
    print(f"Positive samples (T): {positive_count}")

    results = evaluate_model(model, test_loader, device, args.threshold)
    save_results(results, dirs, args.threshold, args)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    args = get_args()
    test(args)
