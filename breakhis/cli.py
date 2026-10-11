"""Import-safe command interface for supported BreaKHis workflows."""
import argparse
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]


def add_data_options(parser):
    parser.add_argument('--dataset',choices=['breakhis'],required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'artifacts/data/manifest.json')
    parser.add_argument('--source',choices=['breakhis_v2','cancer_binary'],default='cancer_binary')
    parser.add_argument('--magnification',choices=['40X','100X','200X','400X','all'],default='40X')
    parser.add_argument('--data-root',type=Path,help='Override this source root after relocating data')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--num-workers',type=int,default=0)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--deterministic',action='store_true')


def main(argv=None):
    parser = argparse.ArgumentParser(description='BreaKHis classification and dataset auditing')
    commands = parser.add_subparsers(dest='command',required=True)
    prepare = commands.add_parser('prepare-data',help='Audit local sources and save patient-separated splits')
    prepare.add_argument('--dataset-root',type=Path,default=ROOT.parent/'Dataset')
    prepare.add_argument('--output',type=Path,default=ROOT/'artifacts/data')
    prepare.add_argument('--seed',type=int,default=0)
    prepare.add_argument('--workers',type=int,default=8)
    for name in ('train','evaluate','prune','finetune'):
        command = commands.add_parser(name)
        add_data_options(command)
        command.add_argument('--output',type=Path,help='Evaluation report path, or checkpoint filename/path inside the parent checkpoint/binary directory')
        if name in ('train','finetune'):
            command.add_argument('--epochs',type=int,default=20)
            command.add_argument('--learning-rate',type=float,default=.001)
        if name=='train':
            command.add_argument('--model',choices=['vgg','resnet','resnet18'],default='vgg')
            command.add_argument('--pretrained',action='store_true',help='Download pretrained ResNet weights')
        else:
            command.add_argument('--checkpoint',type=Path,required=True)
        if name=='prune':
            command.add_argument('--sparsity',type=float,default=.5)
    args = parser.parse_args(argv)
    if args.command=='prepare-data':
        if args.workers<1:
            parser.error('workers must be positive')
        from .manifest import prepare_dataset
        prepare_dataset(args.dataset_root,args.output,args.seed,args.workers)
        return
    if args.batch_size<1 or args.num_workers<0 or getattr(args,'epochs',1)<1:
        parser.error('batch size and epochs must be positive; workers must be nonnegative')
    if args.command=='prune' and not 0<=args.sparsity<=1:
        parser.error('sparsity must be between zero and one')
    from .paths import checkpoint_output
    if args.command=='train':
        try:
            checkpoint_output(args.source,args.magnification,args.model,args.command,args.output)
        except ValueError as error:
            parser.error(str(error))
    import torch
    from torch import nn
    from .data import get_split_loaders
    from .checkpoints import create_model,load_checkpoint,save_checkpoint
    from .training import train,evaluate
    device = torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    loaders = get_split_loaders(args.manifest,args.source,args.magnification,args.data_root,
                                args.batch_size,args.num_workers,device,deterministic=args.deterministic)
    metadata = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    source=args.source,magnification=args.magnification,
                    preprocessing='resize-short-edge-256/crop224/ImageNet-normalization')
    criterion = nn.CrossEntropyLoss()
    if args.command=='train':
        architecture = args.model
        model = create_model(architecture,classes=len(loaders['train'].dataset.classes),
                             pretrained=args.pretrained).to(device)
    else:
        model,payload = load_checkpoint(args.checkpoint,device)
        architecture = payload['architecture']
        if payload['metadata'] != metadata:
            raise ValueError('Checkpoint source, magnification, preprocessing, or manifest differs from selected data')
    print(f'Device: {device}; source: {args.source}; magnification: {args.magnification}',flush=True)
    print('Images:',{split:len(loader.dataset) for split,loader in loaders.items()},flush=True)
    if args.command=='evaluate':
        accuracy,loss = evaluate(model,loaders['test'],criterion)
        report = dict(split='test',accuracy=accuracy,loss=loss,**metadata)
        print(json.dumps(report,indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        return
    try:
        output = checkpoint_output(args.source,args.magnification,architecture,args.command,args.output)
    except ValueError as error:
        parser.error(str(error))
    if args.command=='prune':
        from torch.nn.utils import prune
        for module in model.modules():
            if isinstance(module,(nn.Conv2d,nn.Linear)):
                prune.l1_unstructured(module,'weight',amount=args.sparsity)
        accuracy,loss = evaluate(model,loaders['validation'],criterion)
        save_checkpoint(output,model,architecture,metadata)
        print(f'Validation accuracy after pruning: {accuracy:.4f}; saved {output}')
        return
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,momentum=.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    best_accuracy = float('-inf')
    try:
        for epoch in range(1,args.epochs+1):
            started = time.perf_counter()
            loss = train(model,loaders['train'],criterion,optimizer,epoch=epoch)
            training_seconds = time.perf_counter()-started
            started = time.perf_counter()
            accuracy,val_loss = evaluate(model,loaders['validation'],criterion)
            print(f'Epoch {epoch}: train {training_seconds:.1f}s; validation {time.perf_counter()-started:.1f}s; loss {loss:.4f}; validation loss {val_loss:.4f}; accuracy {accuracy:.4f}',flush=True)
            if accuracy>best_accuracy:
                best_accuracy = accuracy
                save_checkpoint(output,model,architecture,metadata,epoch)
            scheduler.step()
    except KeyboardInterrupt:
        interrupted = output.with_name(output.stem+'-interrupted.pth')
        save_checkpoint(interrupted,model,architecture,metadata)
        print(f'Interrupted; current weights saved to {interrupted}',flush=True)
        return
    print(f'Best validation accuracy: {best_accuracy:.4f}; checkpoint: {output}',flush=True)


if __name__ == '__main__':
    main()
