import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, time
import imageio
from model.MEANet import MEANet
from utils1.data import test_dataset
from skimage import io

# ---------- Metrics Functions ----------
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
    # Simple implementation (structure measure)
    # Ref: https://arxiv.org/abs/1709.06620
    y = gt.mean()
    if y == 0:
        return 1 - pred.mean()
    elif y == 1:
        return pred.mean()
    else:
        alpha = 0.5
        # object-aware
        Q_obj = alpha * np.mean(gt * pred) + (1 - alpha) * np.mean((1 - gt) * (1 - pred))
        # region-aware (rough)
        Q_reg = np.corrcoef(pred.flatten(), gt.flatten())[0, 1]
        return (Q_obj + Q_reg) / 2

def e_measure(pred, gt):
    # Enhanced-alignment measure (simplified version)
    # Ref: https://arxiv.org/abs/1805.10421
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    score = 1 - np.mean((pred - gt) ** 2)
    return score
# --------------------------------------

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
args = parser.parse_args()
opt = args

dataset_path = '/kaggle/input/'

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
    print(dataset)

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

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # save prediction
        imageio.imsave(save_path + name, (res * 255).astype(np.uint8))

        # compute metrics
        mae_list.append(mae(res, gt))
        fm_list.append(f_measure(res, gt))
        sm_list.append(s_measure(res, gt))
        em_list.append(e_measure(res, gt))

        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))

    # final results
    print("Evaluation on {}: ".format(dataset))
    print(" MAE: {:.4f}".format(np.mean(mae_list)))
    print(" F-measure: {:.4f}".format(np.mean(fm_list)))
    print(" S-measure: {:.4f}".format(np.mean(sm_list)))
    print(" E-measure: {:.4f}".format(np.mean(em_list)))
