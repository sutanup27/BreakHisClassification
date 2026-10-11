import contextlib
import importlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

AVAILABLE=importlib.util.find_spec('torch') is not None


class CommandTests(unittest.TestCase):
    def test_cli_requires_dataset_and_rejects_cifar(self):
        for args in [[],['--dataset','cifar10']]:
            result=subprocess.run([sys.executable,'Train.py',*args],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('--dataset',result.stderr)

    def test_all_supported_commands_offer_help(self):
        for command in ['prepare-data','train','evaluate','prune','finetune']:
            result=subprocess.run([sys.executable,'-m','breakhis',command,'--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


@unittest.skipUnless(AVAILABLE,'Requires PyTorch')
class ImportTests(unittest.TestCase):
    def test_imports_do_not_execute_experiments(self):
        import torch
        modules=['Train','Main','Test','PruneTrain','PruneReTrain','PruneEvaluator','PruneViewer',
                 'DataPreview','PruneTest','Testing.PruneTest','Models.VGG',
                 'Utills.PrunUtillCP','Utills.Utill','Utills.ViewerUtills',
                 'breakhis.cli','breakhis.training','breakhis.checkpoints']
        with patch.object(torch,'load',side_effect=AssertionError('Checkpoint loaded on import')):
            with patch('matplotlib.pyplot.show',side_effect=AssertionError('Plot shown on import')):
                with patch('torchvision.datasets.ImageFolder',side_effect=AssertionError('Data loaded on import')):
                    stream=io.StringIO()
                    with contextlib.redirect_stdout(stream):
                        for name in modules: importlib.import_module(name)
                    self.assertEqual(stream.getvalue(),'')

    def test_compatibility_models_and_training_are_shared(self):
        from VGG import VGG as old
        from Models.VGG import VGG as other
        from breakhis.models.vgg import VGG
        from TrainingModules import train
        from Utills.TrainingModulesUtills import train as utility_train
        self.assertIs(old,VGG)
        self.assertIs(other,VGG)
        self.assertIs(train,utility_train)


if __name__=='__main__': unittest.main()
