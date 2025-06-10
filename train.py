
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.ori_vit_seg_modeling import VisionTransformer as ViT_seg_ori
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def self_supervised_distillation(student_outputs, teacher_outputs):
    return torch.nn.functional.mse_loss(student_outputs, teacher_outputs)

def multiple_instance_learning(instance_preds, bag_label):
    return torch.nn.functional.cross_entropy(instance_preds.mean(dim=0, keepdim=True), bag_label)

def train_one_epoch(model, teacher_model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs, bag_label = batch['image'].to(device), batch['label'].to(device)
        student_outputs = model(inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        distill_loss = self_supervised_distillation(student_outputs, teacher_outputs)
        mil_loss = multiple_instance_learning(student_outputs, bag_label)
        loss = distill_loss + mil_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../data/downstream/CVC-ClinicVideoDB/', help='root dir for data')
    parser.add_argument('--dataset', type=str, default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='input patch size')
    parser.add_argument('--seed', type=int, default=9041, help='random seed')
    parser.add_argument('--n_skip', type=int, default=3, help='number of skip-connect')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size')
    parser.add_argument('--pretrained_model_weights', type=str, default='cvc.pth', help='pretrained weights')
    args = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    teacher_model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    teacher_model.load_state_dict(torch.load(args.pretrained_model_weights))
    teacher_model.eval()

    # from your_dataset import get_dataloader
    # train_loader = get_dataloader(args.root_path, args.list_dir, args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(args.max_epochs):
        # loss = train_one_epoch(model, teacher_model, train_loader, optimizer, device)
        # print(f"Epoch {epoch+1}/{args.max_epochs}, Loss: {loss:.4f}")
        pass

    # torch.save(model.state_dict(), "smil_model_final.pth")
