# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import _init_paths
import os
import sys
import numpy as np
import math
import argparse
import pprint
import pdb
import json
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from shutil import copyfile
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    args = parser.parse_args()
    return args

def test():
    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    
    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False
    # image and annotations loading
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    pascal_classes = imdb.classes
    num_classes = imdb.num_classes

    print('{:d} roidb entries'.format(len(roidb)))

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']


    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]

    # data loading
    output_dir = get_output_dir(imdb, save_name)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                             num_classes, training=False, normalize = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                              shuffle=False, num_workers=0,
                              pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
    
    raw_error = [{"tp":0, "fp":0, "tn":0, "fn":0} for _ in range(num_classes)]
    i = 0
    # for each image
    while i < num_images:

        data = next(data_iter)
        with torch.no_grad():
            im_data.resize_(data[0].size()).copy_(data[0])
            im_info.resize_(data[1].size()).copy_(data[1])
            gt_boxes.resize_(data[2].size()).copy_(data[2])
            num_boxes.resize_(data[3].size()).copy_(data[3])

        try:
            det_tic = time.time()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
        except RuntimeError as err:
            # return first index untested
            if "CUDA out of memory" in err.args[0]:
                break
            else:
                print(arr.args[0])
                exit()

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) \
                                * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) \
                                * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        
        # for each class in each image
        for j in xrange(1, num_classes):
            if pascal_classes[j] == "dummy": continue

            inds = torch.nonzero(scores[:,j]>thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array
         
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0 and all_boxes[j][i]:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

        if vis:
            cv2.imwrite('result.png', im2show)

        # calculate precision, recall and F1 for each class
        # roidb[i]['image']       - image file path
        # roidb[i]['boxes']       - all bounding boxes
        # roidb[i]['gt_classes']  - class index for each bounding box
        # all_boxes[1][i]         - all predicted boxes for first class
        
        # get center pxl for each ground truth box
        center_truth = [[] for _ in range(num_classes)]
        box_idx = 0
        for box in roidb[i]['boxes']:
            center_truth[roidb[i]['gt_classes'][box_idx]].append( 
                    (np.average([box[0], box[2]]), np.average([box[1], box[3]])) )
            box_idx+=1

        # center pxl for each pred box
        pred_thresh = 0.4
        center_pred = [[] for _ in range(num_classes)]
        for c in range(1,num_classes):
            for box in all_boxes[c][i]:
                # if score for that box > prediction threshold
                if box[4] >= pred_thresh:
                    center_pred[c].append(
                            (np.average([box[0], box[2]]), np.average([box[1], box[3]])) )

        # for each class except background
        min_dist = 8.5
        for c in range(1, num_classes):
            if pascal_classes[c] == "dummy": continue

            # for predicted box of class c
            uniq_preds = 0
            for prd in center_pred[c]:
                match = False
                # for ground truth box of class c
                for tru in center_truth[c]:
                    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(prd,tru)]))
                    # TP: pred px matches ground truth px only once
                    if dist < min_dist and not match: 
                        raw_error[c]['tp'] += 1
                        uniq_preds+=1
                        match = True
                    # FP: duplicate pred boxes
                    elif dist < min_dist and match:
                        raw_error[c]['fp'] += 1
                
                # FP: no truth box to match pred box
                if not match: 
                    raw_error[c]['fp'] += 1
            
            # FN: if total # ground truths < accurately predicted boxes
            if uniq_preds < len(center_truth[c]):
                raw_error[c]['fn'] += len(center_truth[c]) - uniq_preds
        i+=1

    # total fp, tp, fn
    raw_total = {'tp':0, 'fp':0, 'tn':0, 'fn':0}
    for key in raw_total.keys():
        raw_total[key] = sum(raw_error[c][key] for c in range(1,num_classes))

    raw_error.append(raw_total)

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    # imdb.evaluate_detections(all_boxes, output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))

    # end of testing
    if i == num_images: 
        i = -1
    return i, pascal_classes, raw_error

if __name__ == '__main__':
    # create a backup of test.txt
    test_pth = os.path.join('data','VOCdevkit2007','VOC2007','ImageSets','Main','test.txt')
    test_full_pth = os.path.join('data','VOCdevkit2007','VOC2007','ImageSets','Main','test_full.txt')
    copyfile(test_pth, test_full_pth)

    start_idx = 0
    raw_error = []
    while start_idx >= 0:
        # Edit test.txt to start at start_idx
        with open(test_pth, "w") as f, open(test_full_pth, "r") as f_full:
            # exclude final whitespace
            all_lines = f_full.read().split('\n')[:-1]
            for i in range(start_idx, len(all_lines)):
                f.write(all_lines[i] + "\n")

        crash_idx, classes, raw_error_part = test()
        torch.cuda.empty_cache()

        if crash_idx == -1:
            start_idx = -1
        elif crash_idx == 0:
            start_idx += 1
        else:
            start_idx += crash_idx

        # first overflow
        if not raw_error:
            raw_error[:] = raw_error_part[:]
        # not first overflow
        else:
            for c in range(len(raw_error_part)):
                for key in raw_error_part[c].keys():
                    raw_error[c][key] += raw_error_part[c][key]
    
    # calculate precision, recall, F1 for each class and all classes
    rel_error = [{"prec":0, "recall":0, "f1":0} for _ in range(len(classes))]
    for c in range(1, len(classes)):
        if classes[c] == "dummy" or raw_error[c]['tp'] == 0: continue

        # precision - tp/(tp+fp)
        rel_error[c]['prec'] = raw_error[c]['tp'] / (raw_error[c]['tp']+raw_error[c]['fp'])
        # recall - tp/(tp+fn)
        rel_error[c]['recall'] = raw_error[c]['tp'] / (raw_error[c]['tp']+raw_error[c]['fn'])
        # f1 - 2*[(prec*rec)/(prec+rec)]
        rel_error[c]['f1'] = 2 * \
                ((rel_error[c]['prec'] * rel_error[c]['recall']) / \
                 (rel_error[c]['prec'] + rel_error[c]['recall']))

    # average prec, recall, f1
    rel_total = {"prec":0, "recall":0, "f1":0}
    for key in rel_total.keys():
        rel_total[key] = np.average(
            [rel_error[c][key] for c in range(1, len(classes)) if classes[c] != "dummy"])

    rel_error.append(rel_total)

    with open("output/csvs/error_report.csv","w", newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["----"] + list(classes[1:]) + ["total"])
        for key in raw_error[0].keys():
            writer.writerow([key] + 
                    [raw_error[i][key] for i in range(1,len(raw_error))])
        writer.writerow(['----'])
        for key in rel_error[0].keys():
            writer.writerow([key] + 
                    [rel_error[i][key] for i in range(1,len(rel_error))])
