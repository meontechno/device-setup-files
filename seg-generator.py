import os
import sys
import uuid
from tqdm import tqdm
import argparse

import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.image import load_single_image
from utils.general import Profile, check_img_size, cv2, non_max_suppression
from utils.segment.general import process_mask, process_mask_native
from utils.torch_utils import smart_inference_mode
from utils.logger import logger


COLOR1 = (0, 0, 0)
COLOR2 = (197, 197, 197)
COLOR3 = (255, 255, 255)


def letterbox(im, new_shape=(224, 224), color=COLOR1, auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def get_masked_img_crop(msk, img):
    nonzero_indices = torch.nonzero(msk.unsqueeze(0))
    min_x = torch.min(nonzero_indices[:, 2])
    max_x = torch.max(nonzero_indices[:, 2])
    min_y = torch.min(nonzero_indices[:, 1])
    max_y = torch.max(nonzero_indices[:, 1])

    return img[:, min_y:max_y + 1, min_x:max_x + 1]


def check_and_create_folder(dir_name):
    # Check if the folder exists
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        logger.info(f"Directory '{dir_name}' created.")


def get_images_in_directory(directory):
    files = os.listdir(directory)

    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        logger.error(f"No images found in the directory {directory}")

    return image_files


@smart_inference_mode()
def run(
    model,
    weights=None,
    data_dir=None,
    out_dir="./",
    device=torch.device("cpu")
):
    imgsz = (640, 640)  # inference size (height, width)

    image_files = get_images_in_directory(data_dir)
    img_count = len(image_files)
    
    if img_count < 1:
        return
    
    check_and_create_folder(out_dir)
    logger.info(f"Class:\t{out_dir.rsplit('/')[-1]}")
    logger.info(f"Total images:\t{img_count}")

    # Load model
    stride, _, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    _, _, dt = 0, [], (Profile(), Profile(), Profile())

    for img_file in tqdm(image_files, desc="Generating segments"):

        try:
            frame, img = load_single_image(os.path.join(data_dir, img_file), device=device)
        except Exception as e:
            logger.error(f"Error reading {img_file}")
            continue
        frames_shape = frame.shape[1:]
        frames_tensor_shape = frame.shape

        # Inference
        with dt[1]:
            pred, proto = model(img, augment=False, visualize=False)[:2]

        # NMS
        with dt[2]:
            classes, agnostic_nms = None, None
            conf_thres, iou_thres = 0.60, 0.45
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000, nm=32)

        # logger.info(f"{imgs.shape=}")

        if any([x.shape[0] for x in pred]):
            # Each item in processed_masks corresponds to a single item in a batch
            processed_masks = list()
            pred_items = list()
            total_masks = 0
            total_bboxs = 0

            for i, prd in enumerate(pred):
                total_bboxs += prd.shape[0]
                pre_scale_mask = (process_mask(proto[i], prd[:, 6:], prd[:, :4], img.shape[2:], upsample=True) > 0).float()
                post_scale_mask = torch.nn.functional.interpolate(pre_scale_mask[None], size=frames_shape, mode='bilinear', align_corners=True)[0]
                processed_masks.append(post_scale_mask)

                for c in prd[:, 5].unique():
                    # logger.info(f"{c} in prd")
                    n = (prd[:, 5] == c).sum()
                    pred_items.append({"upc": int(c), "qty": n})

            for _, msk in enumerate(processed_masks):
                total_masks += msk.shape[0]

            masked_imgs = dict()

            for i, msk in enumerate(processed_masks):
                # msk_list = list()
                msk_tensor = None
                for j in range(msk.shape[0]):
                    msk_img_mul = frame * msk[j].unsqueeze(0).expand(frames_tensor_shape)
                    # Change the background color to (197, 197, 197)
                    #msk_img_mul = torch.where(msk_img_mul == 0.0, msk_img_mul + 197.0, msk_img_mul)
                    msk_crop = get_masked_img_crop(msk[j], msk_img_mul)

                    lbox_img, lbox_ratio, lbox_size = letterbox(msk_crop.cpu().permute(1, 2, 0).numpy(), new_shape=(640, 640))
                    if not torch.is_tensor(msk_tensor):
                        msk_tensor = torch.from_numpy(lbox_img).permute(2, 0, 1).unsqueeze(0)
                    else:
                        msk_tensor = torch.cat((msk_tensor, torch.from_numpy(lbox_img).permute(2, 0, 1).unsqueeze(0)), 0)
                masked_imgs[i] = msk_tensor

            for i, mval in masked_imgs.items():
                for j in range(mval.shape[0]):
                    crp_img = mval[j].permute(1, 2, 0).numpy().astype('uint8')
                    # cv2.imshow("test_mask", crp_img)
                    save_path = os.path.join(out_dir, f'{str(uuid.uuid4())}_{i}{j}.jpg')
                    cv2.imwrite(save_path, crp_img)
                    # if cv2.waitKey() == ord('q'):
                    #     pass


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='model path(s)')
    parser.add_argument('--data_dir', type=str, required=True, help='dataset path')
    parser.add_argument('--out_dir', type=str, required=True, help='output dir')
    opt = parser.parse_args()
    return opt


def main(opt):
    data_dir = os.listdir(opt.data_dir)
    abs_dir = [os.path.join(opt.data_dir, x) for x in data_dir]
    out_dirs = [os.path.join(opt.out_dir, x) for x in data_dir]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DetectMultiBackend(opt.weights, device=device, dnn=False, data=None, fp16=False)

    for i in range(len(abs_dir)):
        run(model, data_dir=abs_dir[i], out_dir=out_dirs[i], device=device)
        #run(weights=opt.weights, data_dir=opt.data_dir, out_dir=opt.out_dir)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
