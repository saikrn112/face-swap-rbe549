import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from PRNet.utils.ddfa import ToTensorGjz, NormalizeGjz
import scipy.io as sio
from PRNet.utils.inference import (
    parse_roi_box_from_landmark,
    crop_img,
    predict_68pts,
    predict_dense,
)
from PRNet.utils.cv_plot import plot_kpt
from PRNet.utils.render import get_depths_image, cget_depths_image, cpncc
from PRNEt.utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE = 120

def main(args):
    display = args.display
    debug = args.debug

    out = cv2.VideoWriter('../data/result_del_tri.avi',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          (600, 338))

    # Initialize frontal face detector and shape predictor:
    predictor_model = "../data/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)

    # 1. load pre-tained model
    checkpoint_fp = 'PRNet/models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)[ 'state_dict' ]

    model = getattr(mobilenet_v1, arch)(
        num_classes=62
    )  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    cap = cv2.VideoCapture("../data/sample_video1.gif")
    success = True
    last_frame_pts = []
    while success:
        success , frame = cap.read()
        if len(last_frame_pts) == 0:
            rects = face_detector(frame, 1)
            for rect in rects:
                pts = face_regressor(frame, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                last_frame_pts.append(pts)

        for lmk in last_frame_pts:
            roi_box = parse_roi_box_from_landmark(lmk)
            img = crop_img(frame, roi_box)
            img = cv2.resize(
                img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR
            )
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            pts68 = predict_68pts(param, roi_box)
            lmk[:] = pts68[:2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display images")
    args = parser.parse_args()
    main(args)
