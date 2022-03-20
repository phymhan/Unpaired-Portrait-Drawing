"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import os.path as osp
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torchvision.transforms as transforms
from tqdm import tqdm
import random
from PIL import Image
import cv2
import torch
import numpy as np
from natsort import natsorted

import pdb
st = pdb.set_trace

"""
CUDA_VISIBLE_DEVICES=3 python3 generate_drawing_all.py --dataroot '' --part 3
"""

def to_image(img):
    img = ((img.squeeze().detach().cpu().numpy()+1)/2*255).astype(np.uint8)
    return img

if __name__ == '__main__':
    random.seed(0)
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    imgsize = 512
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.use_mask = False
    opt.use_eye_mask = False
    opt.use_lip_mask = False
    opt.crop_size = imgsize
    opt.load_size = imgsize
    opt.model = 'test_3styles'
    # opt.model = 'test'
    opt.name = 'pretrained'
    opt.epoch = '200'
    opt.imagefolder = 'images2styles'
    opt.no_dropout = True
    opt.output_nc = 1
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    
    if opt.eval:
        model.eval()
    dspth = 'MMVID/data/mmvoxceleb/video'
    respth = 'MMVID/data/mmvoxceleb/draw'
    num_frames = 5
    frame_step = 4
    
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    with open(f"list_videos_4-{opt.part}.txt", 'r') as f:
        videos = [s.strip() for s in f.readlines()]

    for folder_name in tqdm(videos):
        if folder_name.endswith('.txt'):
            continue
        folder_path = os.path.join(dspth, folder_name)
        frames = natsorted(os.listdir(folder_path))
        frames = frames[::frame_step]
        # frames = random.sample(frames, num_frames)
        os.makedirs(osp.join(respth, 'style1', folder_name), exist_ok=True)
        # os.makedirs(osp.join(respth, 'style2', folder_name), exist_ok=True)
        # os.makedirs(osp.join(respth, 'style3', folder_name), exist_ok=True)
        for frame in frames:
            image_path = os.path.join(folder_path, frame)
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((imgsize, imgsize), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()

            data = {'A': img, 'A_paths': image_path}
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            cv2.imwrite(osp.join(respth, 'style1', folder_name, frame), to_image(visuals['fake1']))
            # cv2.imwrite(osp.join(respth, 'style2', folder_name, frame), to_image(visuals['fake2']))
            # cv2.imwrite(osp.join(respth, 'style3', folder_name, frame), to_image(visuals['fake3']))
