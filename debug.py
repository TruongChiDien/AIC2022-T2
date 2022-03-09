import json
import math
import os
import sys
from datetime import datetime
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from config import get_defaultcfgonfig
from models.siamese_baseline import SiameseBaselineModelv1,SiameseLocalandMotionModelBIG
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer, RobertaModel
from collections import OrderedDict
from yacs.config import CfgNode as CN
cfg = CN()
cfg.DATA = CN()
cfg.DATA.CITYFLOW_PATH = "data/AIC21_Track5_NL_Retrieval"
cfg.DATA.TRAIN_JSON_PATH = "data/train.json"
cfg.DATA.EVAL_JSON_PATH = "data/val.json"
cfg.DATA.SIZE = 288
cfg.DATA.CROP_AREA = 1. ## new_w = CROP_AREA * old_w
cfg.DATA.TEST_TRACKS_JSON_PATH = "data/AIC2021-T5-CLV/test_tracks.json"
cfg.DATA.USE_MOTION = True
cfg.DATA.MOTION_PATH = "data/motion_map"
cfg.MODEL = CN()
cfg.MODEL.NAME = "dual-stream"
cfg.MODEL.BERT_TYPE = "ROBERTA"
cfg.MODEL.BERT_NAME = "roberta-large"
cfg.MODEL.IMG_ENCODER = "se_resnext50_32x4d" # se_resnext50_32x4d, efficientnet-b2, efficientnet-b3
cfg.MODEL.NUMcfgLASS = 2498
cfg.MODEL.EMBED_DIM = 1024
cfg.MODEL.car_idloss = False
cfg.MODEL.mo_idloss = False
cfg.MODEL.share_idloss = False
cfg.TRAIN = CN()
cfg.TRAIN.ONE_EPOCH_REPEAT = 30
cfg.TRAIN.EPOCH = 40
cfg.TRAIN.BATCH_SIZE = 32
cfg.TRAIN.NUM_WORKERS = 4
cfg.TRAIN.PRINT_FREQ = 20
cfg.TRAIN.LR = CN()
cfg.TRAIN.LR.BASE_LR = 0.01
cfg.TRAIN.LR.WARMUP_EPOCH = 40
cfg.TRAIN.LR.DELAY = 8
cfg.TEST = CN()
cfg.TEST.RESTORE_FROM = "checkpoints/motion_SE_NOCLS_nonlpaug_288.pth"
cfg.TEST.QUERY_JSON_PATH = "data/AIC2021-T5-CLV/test_queries.json"
cfg.TEST.BATCH_SIZE = 128
cfg.TEST.NUM_WORKERS = 6
cfg.TEST.CONTINUE = ""
best_top1_eval = 0.
cfg = get_defaultcfgonfig()
cfg.merge_from_file(config)
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
usecfguda = True
train_data=CityFlowNLDataset(cfg.DATA, json_path = cfg.DATA.TRAIN_JSON_PATH, transform=transform_test)
trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)
val_data=CityFlowNLDataset(cfg.DATA,json_path = cfg.DATA.EVAL_JSON_PATH, transform=transform_test,Random = False)
valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE*20, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS)
model = SiameseLocalandMotionModelBIG(cfg.MODEL)
model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.TRAIN.LR.BASE_LR)
tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
global_step = 0
best_top1 = 0.
model.train()
batch_time = AverageMeter('Time', ':6.3f')
data_time = AverageMeter('Data', ':6.3f')
losses = AverageMeter('Loss', ':.4e')
top1_acc = AverageMeter('Acc@1', ':6.2f')
top5_acc = AverageMeter('Acc@5', ':6.2f')
progress = ProgressMeter(
    len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT,
    [batch_time, data_time, losses, top1_acc, top5_acc],
    prefix="Epoch: [{}]".format(0))


a = iter(trainloader)
batch = a.next()
image,text,bk,id_car = batch
tokens = tokenizer.batch_encode_plus(text, padding='longest', return_tensors='pt')
pairs,logit_scale,cls_logits = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),image.cuda(),bk.cuda())

