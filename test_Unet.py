import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
from Data import get_data
from Unet_attention import AttentionUNet,UNet,FPNUNet, FPNUNet_LLight ,FPNUNetV2,FPNUNetV3\
    ,UNet,FPNUNetV3_CBAM,FPNUNetV3_CBAM_pro,FPNUNetV3_CCBAM,FPNUNetV3_CBAM_Residual_2\
        ,FPNUNetV3_CBAM_AttnGate,FPNUNet_Light,FPNUNet_Lightt,FPNUNet_Liight,FPNUNet_Lighht\
            ,FPNUNet_ResNetEncoder,FPNUNet_SimpleEncoderFusion,Del_Res,FPNUNet_A_Lightt
            
            

from A import FPNUNetV3_CBAM_Residual_7,FPNUNet_Ligh_7,Del_FPN_F,Del_CBAM,FPNUNetV3_CBAM_Residual_SUM
from medpy.metric.binary import hd95  # pip install medpy
import datetime

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_path = "results/FPNUNetV3_CBAM_Residual_SUM_20250717_1517_48/model_last.pth"



# 加载数据
test_data = get_data(
    image_dir='/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUSI/test/images',
    mask_dir='/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUSI/test/masks'
)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# 加载模型
model = FPNUNetV3_CBAM_Residual_SUM()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(weight_path, map_location=device))
model = model.to(device)
model.eval()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_params, trainable_params = count_parameters(model)
model_name = model.module.__class__.__name__

# ----------------- 评估指标函数 --------------------
def dice_coef(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def precision_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + fp + eps)

def recall_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)

def specificity_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + eps) / (tn + fp + eps)

def pixel_accuracy(pred, target):
    return (pred == target).sum() / np.prod(target.shape)

# ----------------- 测试与统计平均指标 --------------------
dice_list, iou_list, prec_list, recall_list, acc_list, spe_list, hd95_list = [], [], [], [], [], [], []

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        # flatten tensors
        pred_np = preds.cpu().numpy().astype(np.uint8)[0, 0]
        mask_np = masks.cpu().numpy().astype(np.uint8)[0, 0]

        # 计算指标
        dice_list.append(dice_coef(preds, masks).item())
        iou_list.append(iou_score(preds, masks).item())
        prec_list.append(precision_score(preds, masks).item())
        recall_list.append(recall_score(preds, masks).item())
        spe_list.append(specificity_score(preds, masks).item())
        acc_list.append(pixel_accuracy(pred_np, mask_np))
        
        # Hd95 需要预测与标签中至少有一个前景
        try:
            if pred_np.any() and mask_np.any():
                hd95_val = hd95(pred_np, mask_np)
            else:
                hd95_val = np.nan
        except:
            hd95_val = np.nan
        hd95_list.append(hd95_val)

# ----------------- 平均指标结果 --------------------
avg_dice = np.nanmean(dice_list)
avg_iou = np.nanmean(iou_list)
avg_prec = np.nanmean(prec_list)
avg_recall = np.nanmean(recall_list)
avg_spe = np.nanmean(spe_list)
avg_acc = np.nanmean(acc_list)
avg_hd95 = np.nanmean([v for v in hd95_list if not np.isnan(v)])


save_dir = os.path.dirname(weight_path)


os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = os.path.basename(save_dir)
metrics_txt = os.path.join(save_dir, f"{model_name}_metrics_{timestamp}.txt")



results = []

with torch.no_grad():
    for idx, (img, mask) in enumerate(tqdm(test_loader)):
        img = img.to(device)
        mask = mask.to(device)

        output = model(img)
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()

        # 不再计算 metrics，仅记录 sample 名称
        sample_name = f"sample_{idx+1}"
        results.append({"Sample": sample_name})

        # 拼图保存
        comp = torch.cat([img[0], mask[0], pred[0]], dim=2)
        # save_image(comp, os.path.join(save_dir, f"{sample_name}_comp.png"), normalize=True)



print(f"权重路径: {weight_path}")
print(f"保存目录: {save_dir}")
print(f"指标文件路径: {metrics_txt}")
print(f"设备: {device}")

with open(metrics_txt, "w") as f:
    f.write("==== Evaluation Metrics on Test Set ====\n")
    f.write(f"Dice:         {avg_dice:.4f}\n")
    f.write(f"IoU:          {avg_iou:.4f}\n")
    f.write(f"Precision:    {avg_prec:.4f}\n")
    f.write(f"Recall:       {avg_recall:.4f}\n")
    f.write(f"Specificity:  {avg_spe:.4f}\n")
    f.write(f"Accuracy:     {avg_acc:.4f}\n")
    f.write(f"Hd95:         {avg_hd95:.4f}\n")
    f.write("\n==== Model Info ====\n")
    f.write(f"Model Structure:  {model_name}\n")
    f.write(f"Total Params:     {total_params:,}\n")
    f.write(f"Trainable Params: {trainable_params:,}\n")
    f.write("\n==== Inference Hyperparameters ====\n")
    f.write(f"batch_size:       {test_loader.batch_size}\n")
    f.write(f"loss_fn:          0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss\n")
    f.write(f"input_size:       {str(next(iter(test_loader))[0].shape)}\n")

print(f"\n✅ Metrics saved to {metrics_txt}")