import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, time
import imageio
from model.MEANet import MEANet
from utils1.data import test_dataset

# ---------------- Metrics ----------------
def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def f_measure(pred, gt, beta2=0.3):
    pred_bin = (pred >= 0.5).astype(np.uint8)
    tp = (pred_bin * gt).sum()
    prec = tp / (pred_bin.sum() + 1e-8)
    recall = tp / (gt.sum() + 1e-8)
    f = (1 + beta2) * prec * recall / (beta2 * prec + recall + 1e-8)
    return f

def divide_gt(gt):
    h, w = gt.shape
    X, Y = np.where(gt == 1)
    if len(X) == 0 or len(Y) == 0:
        x, y = w // 2, h // 2
    else:
        x, y = int(np.mean(Y)), int(np.mean(X))
    return x, y

def s_object(pred, gt):
    fg = pred * gt
    bg = (1 - pred) * (1 - gt)
    u_fg = np.mean(fg[gt==1]) if np.sum(gt==1) > 0 else 0
    u_bg = np.mean(bg[gt==0]) if np.sum(gt==0) > 0 else 0
    return (u_fg + u_bg) / 2.0

def s_region(pred, gt):
    h, w = gt.shape
    x, y = divide_gt(gt)
    gt1, gt2, gt3, gt4 = gt[:y, :x], gt[:y, x:], gt[y:, :x], gt[y:, x:]
    pr1, pr2, pr3, pr4 = pred[:y, :x], pred[:y, x:], pred[y:, :x], pred[y:, x:]
    def safe_ssim(a,b):
        a = (a - a.min())/(a.max()-a.min()+1e-8)
        b = (b - b.min())/(b.max()-b.min()+1e-8)
        return np.corrcoef(a.flatten(), b.flatten())[0,1] if a.size>0 else 0
    Q1 = safe_ssim(pr1, gt1)
    Q2 = safe_ssim(pr2, gt2)
    Q3 = safe_ssim(pr3, gt3)
    Q4 = safe_ssim(pr4, gt4)
    return (Q1+Q2+Q3+Q4)/4.0

def s_measure(pred, gt, alpha=0.5):
    pred = (pred - pred.min()) / (pred.max()-pred.min()+1e-8)
    gt = (gt>0.5).astype(np.float32)
    if gt.sum() == 0:
        return 1 - pred.mean()
    elif gt.sum() == gt.size:
        return pred.mean()
    else:
        So = s_object(pred, gt)
        Sr = s_region(pred, gt)
        return alpha*So + (1-alpha)*Sr

def e_measure(pred, gt):
    pred = (pred - pred.min())/(pred.max()-pred.min()+1e-8)
    gt = (gt>0.5).astype(np.float32)
    return 1 - np.mean((pred - gt)**2)
# -----------------------------------------

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
args = parser.parse_args()
opt = args

dataset_path = '/kaggle/input/'

# Load MEANet model
model = MEANet()
model.load_state_dict(torch.load('/kaggle/input/meanet/pytorch/default/1/MEANet_EORSSD.pth'))
model.cuda()
model.eval()

test_datasets = ['eorssd']

for dataset in test_datasets:
    save_path = './results/' + 'MEANet-' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/test-images/'
    gt_root = dataset_path + dataset + '/test-labels/'
    print("Testing dataset:", dataset)

    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0

    # metrics accumulators
    mae_list, fm_list, sm_list, em_list = [], [], [], []

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # inference
        time_start = time.time()
        res, s1_sig, edg1, s2, s2_sig, edg2, s3, s3_sig, edg3, s4, s4_sig, edg4, s5, s5_sig, edg5 = model(image)
        time_end = time.time()
        time_sum += (time_end - time_start)

        # resize & normalize
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min())/(res.max()-res.min()+1e-8)

        # save prediction
        imageio.imsave(save_path+name, (res*255).astype(np.uint8))

        # compute metrics
        mae_list.append(mae(res, gt))
        fm_list.append(f_measure(res, gt))
        sm_list.append(s_measure(res, gt))
        em_list.append(e_measure(res, gt))

        if i == test_loader.size-1:
            print('Running time per image: {:.5f}s'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))

    # final dataset results
    print("Evaluation on {}: ".format(dataset))
    print(" MAE       : {:.4f}".format(np.mean(mae_list)))
    print(" F-measure : {:.4f}".format(np.mean(fm_list)))
    print(" S-measure : {:.4f}".format(np.mean(sm_list)))
    print(" E-measure : {:.4f}".format(np.mean(em_list)))
