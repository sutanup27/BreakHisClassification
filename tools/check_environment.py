"""Verify Stage 2 dependencies and execution without training or downloads."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, help='Save environment evidence as JSON')
    args = parser.parse_args()
    packages = ['torch', 'torchvision', 'numpy', 'matplotlib', 'tqdm',
                'pillow', 'scikit-learn', 'torchprofile']
    versions = {name: importlib.metadata.version(name) for name in packages}
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; check the wheel build and driver')
    for name in ['torchvision', 'numpy', 'matplotlib', 'tqdm', 'PIL', 'sklearn', 'torchprofile']:
        importlib.import_module(name)
    for name in ['Train', 'Main', 'DataPreprocessing', 'DataProcess.DataPreprocessing',
                 'TrainingModules', 'VGG', 'Viewer', 'Utill', 'Model_Evaluation']:
        importlib.import_module(name)
    from PIL import Image
    from DataPreprocessing import test_transform
    from VGG import VGG
    device = torch.device(args.device)
    matrix = torch.randn(64, 64, device=device)
    assert torch.isfinite(matrix @ matrix.T).all(), 'Nonfinite tensor computation'
    image = test_transform(Image.new('RGB', (700, 460), (100, 150, 200)))
    model = VGG(classes=2).to(device).eval()
    with torch.inference_mode():
        outputs = model(image.unsqueeze(0).to(device))
    assert tuple(outputs.shape) == (1, 2) and torch.isfinite(outputs).all()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    report = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'executable': sys.executable, 'base_prefix': sys.base_prefix,
        'virtual_environment': sys.prefix != sys.base_prefix,
        'packages': versions, 'device_checked': args.device,
        'cuda_available': torch.cuda.is_available(), 'torch_cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'checks': ['primary imports', 'tensor matmul', 'image preprocessing',
                   'VGG two-class forward', 'pip check'],
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
