import time
import datetime
import numpy as np
import argparse


import torch
from torch import optim, nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.detection import FasterRCNN

from data_helper import LabeledDataset
from helper import collate_fn, draw_box

from models import FeatGEN_CNN, SimCLRResnet
import models, network_helpers
import engine

#image_folder = '../../../proj_files/student_data/data/'
#annotation_csv = '../../../proj_files/student_data/data//annotation.csv'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Auto driving Bounding box train test')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=180, metavar='N',
                        help='number of epochs to train (default: 180)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')

    parser.add_argument('--input-data-images-file', type=str, default='')
    parser.add_argument('--input-data-annotation-file', type=str, default='')
    parser.add_argument('--backbone-pretrained-weights-file', type=str, default='')
    parser.add_argument('--save-final-model-path', type=str, default='')
    args = parser.parse_args()

    image_folder = args.input_data_images_file
    annotation_csv = args.input_data_annotation_file


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Data loading code
    print("Loading data")

    # The scenes from 106 - 133 are labeled
    # Divide the labeled_scene_index into two subsets (training and validation)
    labeled_scene_index_tr = np.arange(106, 129)
    labeled_scene_index_ts = np.arange(129, 134)

    def get_transform():
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                      annotation_file=annotation_csv,
                                      scene_index=labeled_scene_index_tr,
                                      transform=get_transform(),
                                      extra_info=True
                                      )
    train_loader = torch.utils.data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)

    labeled_testset = LabeledDataset(image_folder=image_folder,
                                     annotation_file=annotation_csv,
                                     scene_index=labeled_scene_index_ts,
                                     transform=get_transform(),
                                     extra_info=True
                                     )
    test_loader = torch.utils.data.DataLoader(labeled_testset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)


    print("Creating model")

    # Configs for CombineUpSample
    n_features = 64
    output_size = 3

    model_cnn = FeatGEN_CNN( n_features, output_size)
    model_cnn.to(device)
    params_cnn = [p for p in model_cnn.parameters() if p.requires_grad]

    # F-RCNN
    #model_fnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Initialize Backbone Res-34
    sim_clr = models.simclr_resnet('res34', non_linear_head=False)
    # Load weights for backbone from PIRL pretrained network.
    sim_dict = args.backbone_pretrained_weights_file
    #"../e1_simclr_auto_main_epoch_90"
    sim_clr.load_state_dict(torch.load(sim_dict, map_location=device))

    model_res34 = torchvision.models.resnet34(pretrained=False)

    network_helpers.copy_weights_between_models(sim_clr, model_res34)

    modules = list(model_res34.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 512
    model_fnn = FasterRCNN(backbone=backbone, num_classes=10)


    # ImageNet
    #in_features = model_fnn.roi_heads.box_predictor.cls_score.in_features
    #model_fnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=10)

    model_fnn.to(device)
    params_fnn = [p for p in model_fnn.parameters() if p.requires_grad]

    params_cnn_fnn = list(params_cnn) + list(params_fnn)

    optimizer = optim.SGD(params_cnn_fnn, lr=args.lr, momentum=0.6, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=6e-3)
    # lr_scheduler = CosineAnnealingLR(optimizer, 50, eta_min=1e-3, last_epoch=-1)

    model_save_path = args.save_final_model_path

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        print("Executing Epoch: "+str(epoch))
        engine.train(model_cnn, model_fnn, optimizer, train_loader, device, epoch, save=model_save_path)
        lr_scheduler.step()
        if epoch % 10 == 0:
            print("Testing Epoch: "+str(epoch))
            engine.test(model_cnn,model_fnn,test_loader,device,epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))