# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import numpy as np

from torch.nn import BCEWithLogitsLoss

from sklearn.metrics import f1_score, accuracy_score, average_precision_score

from datasets import UCF101, HMDB51, Kinetics
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from utils import utils
from utils.mil import MILAttention
from utils.meters import TestMeter
from utils.parser import load_config

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import csv
from sklearn.manifold import TSNE


def eval_finetune(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.output_dir}/config.json", "w"), indent=4)
    n = args.n_last_blocks

    # ============ preparing data ... ============
    config = load_config(args)
    config.TEST.NUM_SPATIAL_CROPS = 1
    if args.dataset == "Endoscapes":
        dataset_train = Endoscapes(cfg=config, mode="train", num_retries=10)
        dataset_val = Endoscapes(cfg=config, mode="val", num_retries=10)
        dataset_test = Endoscapes(cfg=config, mode="test", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(  
        dataset_test,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Data loaded with {len(dataset_train)} train, {len(dataset_val)} and {len(dataset_test)} val imgs.")

    # ============ building network ... ============
    if config.DATA.USE_FLOW or config.MODEL.TWO_TOKEN:
        model = get_aux_token_vit(cfg=config, no_head=True)
        model_embed_dim = 2 * model.embed_dim
    else:
        if args.arch == "vit_base":
            model = get_vit_base_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
        elif args.arch == "swin":
            model = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
            model_embed_dim = 1024
        else:
            raise Exception(f"invalid model: {args.arch}")

    if not args.scratch and args.pretrained_weights:
        ckpt = torch.load(args.pretrained_weights, map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = model.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded model with msg: {msg}")
    elif args.scratch:
        ckpt = torch.load('kinetics400_vitb_ssl.pth', map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = model.load_state_dict(renamed_checkpoint, strict=False)
        print(f"Loaded model with msg: {msg}")

    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")

    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.

    # Optionally resume from a checkpoint
 
    best_accuracy = 0.0
    best_map = 0.0
    best_f1 = 0.0
    best_accuracy_epoch = 0
    best_map_epoch = 0
    best_f1_epoch = 0

    # set optimizer
    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'lr': scaled_lr},
         {'params': linear_classifier.parameters(), 'lr': scaled_lr}],
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    to_restore = {"epoch": 0, "best_acc": 0., "best_f1": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    # best_f1 = to_restore["best_acc"]
    best_f1 = to_restore["best_f1"]#T:上面的换成这个，怀疑一开始的best_acc是不对的，12.15

    # 初始化历史最佳指标
    best_f1, best_accuracy, best_map = 0.0, 0.0, 0.0
    best_f1_epoch, best_accuracy_epoch, best_map_epoch = 0, 0, 0

    for epoch in range(start_epoch, args.epochs):
        train_stats = train(args, model, linear_classifier, optimizer, 
                            train_loader, epoch, n, args.avgpool_patchtokens)
        
        # 默认先不保存结果
        save_results = False
        improved_metrics = []  # 记录哪几个指标得到提升
        
        # 进行验证
        test_stats, f1, accuracy, map_score = validate_network(
            val_loader, model, linear_classifier, n, args.avgpool_patchtokens, 
            epoch=epoch, save_results=False)  # 默认False，不输出文件
        
        to_restore["best_acc"] = best_accuracy
        to_restore["best_map"] = best_map
        to_restore["best_f1"] = best_f1
        if utils.is_main_process():
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "backbone_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
                "best_acc": best_accuracy,#T: 增加保存权重时的best_acc和best_map，12.15；
                "best_map": best_map,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

        best_f1 = max(best_f1, f1)
        # print(f'Max F1 score so far: {best_f1 * 100:.1f}%')
        #T：新增下面的代码，12.13
        # 获取当前轮次的 Accuracy 和 mAP
        accuracy = test_stats.get("accuracy", 0.0)  # 假设 validate_network 返回了 Accuracy
        map_score = test_stats.get("mAP", 0.0)     # 假设 validate_network 返回了 mAP
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_epoch = epoch

        if map_score > best_map:
            best_map = map_score
            best_map_epoch = epoch
        
        # 更新日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'f1': f1,
            'accuracy': accuracy,
            'map': map_score,
            'best_f1': best_f1,
            'best_accuracy': best_accuracy,
            'best_map': best_map,
        }
        
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
                
    test_results_path = os.path.join(args.output_dir, "test_results.txt")
    with open(test_results_path, 'w') as f:
        f.write("============ Test Results Summary ============\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Data Path: {args.data_path}\n")
        f.write(f"Model Architecture: {args.arch}\n")
        f.write("\nPerformance Metrics:\n")
        f.write(f"F1 Score: {test_f1*100:.2f}%\n")
        f.write(f"Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"mAP Score: {test_map*100:.2f}%\n")
        f.write("\nBest Training Results:\n")
        f.write(f"Best F1: {best_f1*100:.2f}% (epoch {best_f1_epoch})\n")
        f.write(f"Best Accuracy: {best_accuracy*100:.2f}% (epoch {best_accuracy_epoch})\n")
        f.write(f"Best mAP: {best_map*100:.2f}% (epoch {best_map_epoch})\n")
        
    print(f"\nDetailed test results have been saved to: {test_results_path}")
    print("==============================================")


def train(args, model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    criterion = BCEWithLogitsLoss()#T:更改损失函数
    model.train()
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)


    for (inp, target, sample_idx, meta) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True).float()#T: 更改上述代码为此

        # 前向传播
        with torch.cuda.amp.autocast():
            intermediate_output = model(inp)
            logits, attention_weights = linear_classifier(intermediate_output)
            
            # 计算损失
            loss = criterion(logits, target)

         # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录指标
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool, is_test=False, epoch=None, save_results=False):
    model.eval()
    linear_classifier.eval()

    torch.cuda.empty_cache()  # 强制释放显存碎片

    features, labels = [], []
    all_preds, all_labels, all_scores = [], [], []

    mode = "test" if is_test else f"val_epoch_{epoch}"
    vis_dir = os.path.join(args.output_dir, 'tsne_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    predictions_dir = os.path.join(args.output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    for inp, target, _, meta in val_loader:
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        intermediate_output = model(inp)
        output, attention_weights = linear_classifier(intermediate_output)

        features.append(intermediate_output.cpu().numpy())
        labels.append(target.cpu().numpy())

        preds = torch.sigmoid(output) > 0.5
        scores = torch.sigmoid(output)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # 指标计算
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels.ravel(), all_preds.ravel())
    map_score = average_precision_score(all_labels, all_scores, average='macro')

        for class_idx in range(num_classes):
            indices = labels[:, class_idx] == 1
            plt.scatter(tsne_results[indices, 0],
                        tsne_results[indices, 1],
                        label=f'Class {class_idx + 1}',
                        alpha=0.6,
                        color=colors[class_idx])

        plt.legend()
        plt.title(f't-SNE Visualization ({mode})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

        tsne_save_path = os.path.join(vis_dir, f'tsne_{mode}.png')
        plt.savefig(tsne_save_path, dpi=300)
        plt.close()

        # 预测结果保存为 CSV 和 TXT（保持原有逻辑）
        csv_save_path = os.path.join(predictions_dir, f'predictions_{mode}.csv')
        with open(csv_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Sample Index', 'True Label 1', 'True Label 2', 'True Label 3',
                            'Predicted Label 1', 'Predicted Label 2', 'Predicted Label 3',
                            'Score 1', 'Score 2', 'Score 3'])
            for idx, (true_label, pred_label, score) in enumerate(zip(all_labels, all_preds, all_scores)):
                writer.writerow([idx] + true_label.tolist() + pred_label.tolist() + score.tolist())

        txt_save_path = os.path.join(predictions_dir, f'predictions_{mode}.txt')
        with open(txt_save_path, mode='w') as file:
            for idx, (true_label, pred_label, score) in enumerate(zip(all_labels, all_preds, all_scores)):
                file.write(f'Sample {idx}:\n')
                file.write(f'  True labels: {true_label.tolist()}\n')
                file.write(f'  Predicted labels: {pred_label.tolist()}\n')
                file.write(f'  Scores: {score.tolist()}\n\n')

    return {'accuracy': accuracy, 'mAP': map_score, 'f1': f1}, f1, accuracy, map_score
 



class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=3):
        super().__init__()
        feature_dim = 256
        self.projection = nn.Linear(768, feature_dim)
        self.mil_attention = MILAttention(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, x):
        if x.dim() == 2:
            B, D = x.shape
            assert D == 768, f"Expected input dimension 768, got {D}"
            
        x = self.projection(x)
        x = x.unsqueeze(1)
        
        context, att_weights = self.mil_attention(x)
        logits = self.classifier(context)
        return logits, att_weights


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
   # 基础参数
    parser.add_argument('--n_last_blocks', default=4, type=int,
                      help='Number of last blocks to use')
    parser.add_argument('--avgpool_patchtokens', action='store_true',
                      help="""Whether to average all tokens or just the CLS token.""")
    # 模型架构参数 - 删除重复的arch参数定义
    parser.add_argument('--patch_size', default=16, type=int, 
                      help='Patch resolution of the model.')
    parser.add_argument('--test_only', action='store_true', help='Run only test set without training/validation.')
    
    # MIL相关参数
    parser.add_argument('--mil_hidden_dim', default=512, type=int,
                      help='Hidden dimension of MIL attention')
    parser.add_argument('--mil_dropout', default=0.1, type=float, 
                      help='Dropout rate in MIL attention')
    
    # 预训练和检查点参数
    parser.add_argument('--pretrained_weights', default='', type=str,
                      help="Path to pretrained weights to evaluate.")
    parser.add_argument('--lc_pretrained_weights', default='', type=str,
                      help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                      help='Key to use in the checkpoint (example: "teacher")')
    
    # 训练相关参数
    parser.add_argument('--epochs', default=100, type=int,
                      help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")

    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained_model_weights', default='polypdiag.pth', type=str, help='pre-trained weights')

    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    eval_finetune(args)
