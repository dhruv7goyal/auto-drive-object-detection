import math
import os
import sys
from shapely.geometry import Polygon

import torch

import utils


def train(model_cnn, model, optimizer, data_loader, device, epoch, save):

    model_cnn.train()
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, 8, header):

        images = torch.stack(images)
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # Returns output -> convoluted and concatenated image in shape [3x800x800]
        output = model_cnn(images)

        loss_dict = model(output, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if save and (epoch % 10) == 0:
        print("Saving Model for epoch: "+str(epoch))
        torch.save(model_cnn.state_dict(), os.path.join(save,"/cnn", 'epoch-{}.pth'.format(epoch)))
        torch.save(model.state_dict(), os.path.join(save, "/frcnn" , 'epoch-{}.pth'.format(epoch)))


def test(model_cnn, model, data_loader, device,epoch):
    model_cnn.eval()
    model.eval()

    total = 0
    total_ats_bounding_boxes = 0
    total_ts_road_map = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            total += 1
            sample, target = data
            sample = torch.stack(sample)
            sample = sample.to(device)
            output_fgencnn = model_cnn(sample)
            output = model(output_fgencnn)
            out, tar = get_bounding_boxes(output,target)

            ats_bounding_boxes = compute_ats_bounding_boxes(out[0].cpu(),tar[0])

            ts_road_map = 0
            total_ats_bounding_boxes += ats_bounding_boxes
            total_ts_road_map += ts_road_map

    print('TOTAL Bounding Box Score: {} - Road Map Score: {}'.format(
        total_ats_bounding_boxes / total, total_ts_road_map / total
    ))    


def get_bounding_boxes(pred,target):
    pred_list = []
    tar_list = []
    for t in pred:
        a = torch.stack([t['boxes'][:,0],t['boxes'][:,0],t['boxes'][:,2],t['boxes'][:,2]], dim=1).to('cpu')
        b = torch.stack([t['boxes'][:,1],t['boxes'][:,3],t['boxes'][:,1],t['boxes'][:,3]], dim=1).to('cpu')
        pred_list.append(torch.stack([a, b], dim=1).to('cpu'))

    for t in target:
        a = torch.stack([t['boxes'][:,0],t['boxes'][:,0],t['boxes'][:,2],t['boxes'][:,2]], dim=1)
        b = torch.stack([t['boxes'][:,1],t['boxes'][:,3],t['boxes'][:,1],t['boxes'][:,3]], dim=1)
        tar_list.append(torch.stack([a, b], dim=1).to('cpu'))

    return tuple(pred_list),tuple(tar_list)


def compute_ats_bounding_boxes(boxes1, boxes2):
    num_boxes1 = boxes1.size(0)
    num_boxes2 = boxes2.size(0)

    boxes1_max_x = boxes1[:, 0].max(dim=1)[0]
    boxes1_min_x = boxes1[:, 0].min(dim=1)[0]
    boxes1_max_y = boxes1[:, 1].max(dim=1)[0]
    boxes1_min_y = boxes1[:, 1].min(dim=1)[0]

    boxes2_max_x = boxes2[:, 0].max(dim=1)[0]
    boxes2_min_x = boxes2[:, 0].min(dim=1)[0]
    boxes2_max_y = boxes2[:, 1].max(dim=1)[0]
    boxes2_min_y = boxes2[:, 1].min(dim=1)[0]

    condition1_matrix = (boxes1_max_x.unsqueeze(1) > boxes2_min_x.unsqueeze(0))
    condition2_matrix = (boxes1_min_x.unsqueeze(1) < boxes2_max_x.unsqueeze(0))
    condition3_matrix = (boxes1_max_y.unsqueeze(1) > boxes2_min_y.unsqueeze(0))
    condition4_matrix = (boxes1_min_y.unsqueeze(1) < boxes2_max_y.unsqueeze(0))
    condition_matrix = condition1_matrix * condition2_matrix * condition3_matrix * condition4_matrix

    iou_matrix = torch.zeros(num_boxes1, num_boxes2)
    for i in range(num_boxes1):
        for j in range(num_boxes2):
            if condition_matrix[i][j]:
                iou_matrix[i][j] = compute_iou(boxes1[i], boxes2[j])

    iou_max = iou_matrix.max(dim=0)[0]

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    total_threat_score = 0
    total_weight = 0
    for threshold in iou_thresholds:
        tp = (iou_max > threshold).sum()
        threat_score = tp * 1.0 / (num_boxes1 + num_boxes2 - tp)
        total_threat_score += 1.0 / threshold * threat_score
        total_weight += 1.0 / threshold

    average_threat_score = total_threat_score / total_weight

    return average_threat_score


def compute_iou(box1, box2):
    a = Polygon(torch.t(box1)).convex_hull
    b = Polygon(torch.t(box2)).convex_hull

    return a.intersection(b).area / a.union(b).area

