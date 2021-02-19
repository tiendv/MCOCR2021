import os
import cv2
import copy
import time
import json
import torch
import logging
import argparse
import numpy as np
from PIL import Image

from timm.models import create_model, apply_test_time_pool
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging

RETURN_ORIENTATION = [
    'None',
    'ROTATE_90_COUNTERCLOCKWISE',
    'ROTATE_180',
    'ROTATE_90_CLOCKWISE',
]

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientnet_b7',
                    help='model architecture (default: tf_efficientnet_b7)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=4,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='rotated_receipt_90_180/effnet_b7_rotate.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')

setup_default_logging()
args = parser.parse_args()
# might as well try to do something useful...
args.pretrained = args.pretrained or not args.checkpoint

# create model
model = create_model(
    args.model,
    num_classes=args.num_classes,
    in_chans=3,
    pretrained=args.pretrained,
    checkpoint_path=args.checkpoint)

_logger.info('Model %s created, param count: %d' %
                (args.model, sum([m.numel() for m in model.parameters()])))

config = resolve_data_config(vars(args), model=model)
model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)

stream = torch.cuda.Stream()
mean = torch.tensor([x * 255 for x in config['mean']]).cuda().view(1, 3, 1, 1)
std = torch.tensor([x * 255 for x in config['std']]).cuda().view(1, 3, 1, 1)

if args.num_gpu > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
else:
    model = model.cuda()

def rotate_dir(dataset_dir, output_dir):

    loader = create_loader(
        Dataset(dataset_dir),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    topk_prob = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            try:
                input = input.cuda()
                labels = model(input)
                topk = labels.topk(k)[0]
                topk = topk.cpu().numpy()
                # print(topk)
                topk = np.exp(topk) / np.sum(np.exp(topk), axis=-1)[:, np.newaxis]
                # print(topk)
                topk_prob.append(topk)
                topk = labels.topk(k)[1]
                topk_ids.append(topk.cpu().numpy())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_freq == 0:
                    _logger.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time))
            except Exception:
                pass

    topk_ids = np.concatenate(topk_ids, axis=0).squeeze()
    topk_prob = np.concatenate(topk_prob, axis=0).squeeze()
    
    # print(topk_ids)
    # print(topk_prob)

    os.makedirs(os.path.join(output_dir, 'final'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'debug'), exist_ok=True)

    print('START ROTATING')
    
    filenames = loader.dataset.filenames(basename=True)
    for filename, label, prob in zip(filenames, topk_ids, topk_prob):
        inp_path = os.path.join(dataset_dir, filename)
        out_final_path = os.path.join(output_dir, 'final', filename)
        out_debug_path = os.path.join(output_dir, 'debug', filename)
        
        img_rotate = img = cv2.imread(inp_path)
        if label[0] == 1: img_rotate = cv2.rotate(img_rotate, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if label[0] == 2: img_rotate = cv2.rotate(img_rotate, cv2.ROTATE_180)
        if label[0] == 3: img_rotate = cv2.rotate(img_rotate, cv2.ROTATE_90_CLOCKWISE)
        
        cv2.imwrite(out_final_path, img_rotate)

        if label[0] != 0:
            img_debug = np.zeros(shape=(max(img.shape[0], img_rotate.shape[0]), img.shape[1] + img_rotate.shape[1], 3), dtype=np.float32)
            img_debug[:img.shape[0], :img.shape[1]] = img
            img_debug[:img_rotate.shape[0], img.shape[1]:] = img_rotate
            cv2.imwrite(out_debug_path, img_debug)
            
    print('DONE ROTATING')


def rotate_img(img):
    model.eval()

    end = time.time()
    
    img = torch.from_numpy(img.reshape(1, *img.shape).transpose((0,3,1,2)))
    with torch.cuda.stream(stream):
        img = img.cuda(non_blocking=True)
        img = img.float().sub_(mean).div_(std)    # Normalize Mean Variance
        img = img.cuda()
    
    with torch.no_grad():
        labels = model(img).cpu().numpy()
    labels = np.exp(labels) / np.sum(np.exp(labels))
    res = "{}\n{}".format(RETURN_ORIENTATION[np.argmax(labels)], np.amax(labels))
    
    _logger.info('Predicted image with label = {}. Time {:.3f}'.format(res, time.time()-end))
    
    return res
if __name__ == '__main__':
    img_path = '20201116_164641.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, config['input_size'][1:])
    print(config['input_size'][1:])
    print(rotate_img(img))