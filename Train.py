"""Primary BreaKHis training entry point."""
import argparse
from pathlib import Path


def main(default_model="vgg"):
    parser = argparse.ArgumentParser(description="Train a BreaKHis classifier")
    parser.add_argument("--dataset", required=True, choices=["breakhis"])
    parser.add_argument("--magnification", choices=["40X", "100X", "200X", "400X"], default="40X")
    parser.add_argument("--model", choices=["vgg", "resnet"], default=default_model)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "Dataset" / "BreaKHis_v2")
    args = parser.parse_args()

    import torch
    from torch import nn
    from torchvision import models
    from DataPreprocessing import get_class_names, get_dataloaders
    from TrainingModules import Training, evaluate
    from VGG import VGG
    from Viewer import plot_accuracy, plot_loss

    train_loader, test_loader = get_dataloaders(
        args.data_root / args.magnification, batch_size=args.batch_size, dataset=args.dataset,
    )
    class_names = get_class_names(train_loader.dataset)
    num_classes = len(class_names)
    print(f"Dataset: {args.dataset}; labels: {class_names}; classes: {num_classes}")
    if args.model == "vgg":
        model = VGG(classes=num_classes)
    else:
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_model, losses, test_losses, accs = Training(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=args.epochs, scheduler=scheduler,
    )
    metric, _ = evaluate(best_model, test_loader)
    output_dir = Path(__file__).resolve().parent / "checkpoint" / args.magnification / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model, output_dir / f"{args.model}_{metric}.pth")
    plot_accuracy(accs)
    plot_loss(losses, test_losses)


if __name__ == "__main__":
    main()
