import time
import cv2
import glob
import os
import viz
import torch
import yaml
import argparse
import demo_utils
import numpy as np
import noise_image
from thop import profile
from src import get_model_cfg
from torch.utils.data import Dataset
from src.models import TopicFM
from src.utils.dataset import read_img_gray
from sklearn.metrics import mean_squared_error
from src.config.default import get_cfg_defaults
# from tqdm import tqdm
# from configs.megadepth_test import cfg as megadepth_cfg
# from configs.scannet_test import cfg as scannet_cfg


def get_model_config(method_name, dataset_name, root_dir='viz'):
    config_file = f'{root_dir}/configs/{method_name}.yml'
    with open(config_file, 'r') as f:
        model_conf = yaml.load(f, Loader=yaml.FullLoader)[dataset_name]
    return model_conf


class DemoDataset(Dataset):
    def __init__(self, dataset_dir, img_file=None, resize=0, down_factor=16):
        self.dataset_dir = dataset_dir
        if img_file is None:
            self.list_img_files = glob.glob(os.path.join(dataset_dir, "*.*"))
            self.list_img_files.sort()
        else:
            with open(img_file) as f:
                self.list_img_files = [os.path.join(dataset_dir, img_file.strip()) for img_file in f.readlines()]
        self.resize = resize
        self.down_factor = down_factor

    def __len__(self):
        return len(self.list_img_files)

    def __getitem__(self, idx):
        img_path = self.list_img_files[idx]  # os.path.join(self.dataset_dir, self.list_img_files[idx])
        img, scale = read_img_gray(img_path, resize=self.resize, down_factor=self.down_factor)
        return {"img": img, "id": idx, "img_path": img_path}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize matches')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--method', type=str, default="topicfmv2")
    parser.add_argument('--dataset_dir', type=str, default='data/aachen-day-night')
    parser.add_argument('--pair_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, choices=['megadepth', 'scannet', 'aachen_v1.1', 'inloc'],
                        default='scannet')
    parser.add_argument('--config_file', type=str, default="configs/scannet_test_topicfmfast.py")
    parser.add_argument('--measure_time', action="store_true", default=True)
    parser.add_argument('--no_viz', action="store_true", default=True)
    parser.add_argument('--ckpt', type=str, default='./pretrained/topicfm_plus.ckpt')
    # parser.add_argument('--compute_eval_metrics', action="store_true")
    parser.add_argument('--run_demo', action="store_true")

    args = parser.parse_args()

    model_cfg = get_model_config(args.method, args.dataset_name)
    class_name = model_cfg["class"]
    # model = viz.__dict__[class_name](model_cfg)
    # TopicFM_P = viz.__dict__[class_name](model_cfg)
    data_cfg = get_cfg_defaults()

    # ------------------------Load model----------------------------

    conf = dict(get_model_cfg())
    conf['match_coarse']['thr'] = model_cfg['match_threshold']
    for k, v in model_cfg['coarse_model_cfg'].items():
        conf["coarse"][k] = v
    conf['loss']['fine_type'] = "sym_epi"
    # print("model config: ", conf)

    data_cfg.merge_from_file(args.config_file)
    TopicFMv2 = TopicFM(config=conf)
    ckpt_dict = torch.load(args.ckpt)
    TopicFMv2.load_state_dict(ckpt_dict['state_dict'])
    TopicFMv2.eval().cuda()

    # ----------------------------------读取彩色图像------------------------------
    img0_path = "./assets/1DSMsets/pair7-2.png"
    img1_path = "./assets/1DSMsets/pair7-1.png"
    output_path = "./outputs/1DSMsets/pair007.jpg"

    # --------------------Additive noise image ------------------
    # img0 = noise_image.Additive_noise(img0_path, 0)
    # output_path = "./output/1DSMsets/pair1+snr0.jpg"

    # --------------------stripe noise image --------------------
    # img0 = noise_image.stripe_noise(img0_path, 0.1)
    # output_path = "./output/1DSMsets/pair1+0p10.jpg"


    img0 = cv2.imread(img0_path)
    img0 = demo_utils.resize(img0, 512)

    img1 = cv2.imread(img1_path)
    img1 = demo_utils.resize(img1, 512)

    img0_g = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    img1_g = cv2.imread(img1_path, 0)
    img1_g = demo_utils.resize(img1_g, 512)

    data = {'image0': torch.from_numpy(img0_g / 255.)[None, None].cuda().float(),
            'image1': torch.from_numpy(img1_g / 255.)[None, None].cuda().float()}

    tic = time.time()
    with torch.no_grad():
        TopicFMv2(data)

    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()

    # --------------------------RANSAC Outlier Removal----------------------------------
    # F_hat, mask_F = cv2.findFundamentalMat(kpts0, kpts1, method=cv2.USAC_ACCURATE,
    #                                        ransacReprojThreshold=1, confidence=0.99)
    F_hat, mask_F = cv2.findFundamentalMat(kpts0, kpts1, method=cv2.USAC_MAGSAC,
                                           ransacReprojThreshold=1, confidence=0.99)

    toc = time.time()
    tt1 = toc - tic
    if mask_F is not None:
        mask_F = mask_F[:, 0].astype(bool)
    else:
        mask_F = np.zeros_like(kpts0[:, 0]).astype(bool)

    # display = demo_utils.draw_match(img0, img1, kpts0, kpts1)
    display = demo_utils.draw_match(img0, img1, kpts0[mask_F], kpts1[mask_F])

    # --------------------------------------------------------------------------------------
    putative_num = len(kpts0)
    correct_num = len(kpts0[mask_F])
    inliner_ratio = correct_num / putative_num
    text1 = "putative_num:{}".format(putative_num)
    text2 = 'correct_num:{}'.format(correct_num)
    text3 = 'inliner ratio:%.3f' % inliner_ratio
    text4 = 'run time: %.3fs' % tt1

    print('putative_num:{}'.format(putative_num), '\ncorrect_num:{}'.format(correct_num),
          '\ninliner ratio:%.3f' % inliner_ratio, '\nrun time: %.3fs' % tt1)

    cv2.putText(display, str(text1), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text2), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text3), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text4), (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(output_path, display)

    flops, params = profile(TopicFMv2, inputs=(data,))
    print("参数量：", "%.2f" % (params / (1000 ** 2)), "M")
    print("GFLOPS：", "%.2f" % (flops / (1000 ** 3)))

