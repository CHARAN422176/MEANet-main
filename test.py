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

def s_measure(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    alpha = 0.5
    # object-aware similarity
    fg = pred[gt == 1]
    bg = pred[gt == 0]
    o_fg = np.mean(fg) if fg.size > 0 else 0
    o_bg = np.mean(bg) if bg.size > 0 else 0
    object_score = alpha * o_fg + (1 - alpha) * (1 - o_bg)
    # region-aware similarity (divide into 4 regions)
    h, w = gt.shape
    y, x = h // 2, w // 2
    gt_quads = [gt[:y, :x], gt[:y, x:], gt[y:, :x], gt[y:, x:]]
    pr_quads = [pred[:y, :x], pred[:y, x:], pred[y:, :x], pred[y:, x:]]
    region_score = 0
    for gq, pq in zip(gt_quads, pr_quads):
        region_score += np.mean(1 - np.abs(pq - gq))
    region_score /= 4.0
    return 0.5 * (object_score + region_score)

def e_measure(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    fm = np.mean(pred)
    gt_mean = np.mean(gt)
    align_matrix = 2 * (pred - fm) * (gt - gt_mean) / (
        (pred - fm) ** 2 + (gt - gt_mean) ** 2 + 1e-8
    )
    return np.mean((align_matrix + 1) ** 2 / 4)
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

        # save prediction as grayscale uint8
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
