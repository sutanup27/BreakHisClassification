"""Check dataset contracts without importing optional training dependencies.

Execute the loader functions with small dataset doubles; actual torchvision
loading and tensor computations require a separate installed-environment smoke test.
"""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Folder:
    classes = ["benign", "malignant"]

    def __init__(self, root, transform=None):
        self.classes = list(type(self).classes)

    def __len__(self):
        return 10


class Subset:
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices


def split(dataset, sizes):
    return Subset(dataset, range(sizes[0])), Subset(dataset, range(sizes[0], sum(sizes)))


def loader_functions(path, extra=None):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(train_transform=None, test_transform=None,
                     ImageFolder=Folder, Subset=Subset, random_split=split)
    namespace.update(extra or {})
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class WorkflowSeparationTests(unittest.TestCase):
    def setUp(self):
        self.api = loader_functions(ROOT / "DataPreprocessing.py")

    def test_primary_labels_derive_two_classes(self):
        train, test = self.api["get_datasets"]("unused", dataset="breakhis")
        self.assertEqual(self.api["get_class_names"](train), ["benign", "malignant"])
        self.assertEqual(len(self.api["get_class_names"](test)), 2)

    def test_primary_rejects_cifar_before_loading(self):
        with self.assertRaisesRegex(ValueError, "breakhis"):
            self.api["get_datasets"]("unused", dataset="cifar10")

    def test_primary_rejects_unexpected_class_folders(self):
        class WrongFolder(Folder):
            classes = ["airplane", "car"]
        self.api["ImageFolder"] = WrongFolder
        with self.assertRaisesRegex(ValueError, "benign.*malignant"):
            self.api["get_datasets"]("unused", dataset="breakhis")

    def test_nested_split_labels(self):
        nested = Subset(Subset(Folder("unused"), [0]), [0])
        self.assertEqual(len(self.api["get_class_names"](nested)), 2)

    def test_cifar_requires_explicit_selection_and_download_opt_in(self):
        calls = []
        def cifar(**kwargs):
            calls.append(kwargs)
            return object()
        api = loader_functions(ROOT / "experiments/cifar10/DataPreprocessing.py", {"CIFAR10": cifar})
        with self.assertRaises(TypeError):
            api["get_datasets"]("unused")
        with self.assertRaises(ValueError):
            api["get_datasets"]("unused", dataset="breakhis")
        api["get_datasets"]("unused", dataset="cifar10")
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["download"] is False for call in calls))

    def test_cli_requires_dataset_and_rejects_cifar(self):
        for args in [[], ["--dataset", "cifar10"]]:
            result = subprocess.run([sys.executable, str(ROOT / "Train.py"), *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--dataset", result.stderr)

    def test_cifar_dataloader_forwards_dataset_selection(self):
        class Loader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
        api = loader_functions(ROOT / "experiments/cifar10/DataPreprocessing.py",
                               {"CIFAR10": lambda **kwargs: Folder("unused"),
                                "DataLoader": Loader})
        api["set_seed"] = lambda *args, **kwargs: None
        train, test = api["get_dataloaders"]("unused", dataset="cifar10", num_workers=0)
        self.assertIsInstance(train.dataset, Folder)
        self.assertIsInstance(test.dataset, Folder)

    def test_classifier_uses_discovered_label_count(self):
        tree = ast.parse((ROOT / "Train.py").read_text())
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        self.assertTrue(any(
            any(isinstance(t, ast.Name) and t.id == "num_classes" for t in n.targets)
            and ast.unparse(n.value) == "len(class_names)" for n in assignments
        ))
        source = (ROOT / "Train.py").read_text()
        self.assertIn("VGG(classes=num_classes)", source)
        self.assertIn("nn.Linear(model.fc.in_features, num_classes)", source)


if __name__ == "__main__":
    unittest.main()
