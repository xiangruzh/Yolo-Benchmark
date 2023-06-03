import torch
import numpy as np


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

prediction = np.load('000000.npy')
prediction = torch.from_numpy(prediction)

B, abC, H, W = prediction.size()
KA = 5
NC = 20
prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, -1, abC)

conf_pred = prediction[..., :KA].contiguous().view(B, -1, 1)
cls_pred = prediction[..., 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)
txtytwth_pred = prediction[..., (1+NC)*KA:].contiguous().view(B, -1, 4)

conf_pred = conf_pred[0]            #[H*W*KA, 1]
cls_pred = cls_pred[0]              #[H*W*KA, NC]
txtytwth_pred = txtytwth_pred[0]    #[H*W*KA, 4]

max_conf, max_conf_idx = torch.max(conf_pred, dim=0)

max_conf_cls = cls_pred[max_conf_idx]
_, cls_result = torch.max(max_conf_cls, dim=1)

print('Max_Confidence: {}'.format(max_conf.numpy()))
print('Max_Confidence index: {}'.format(max_conf_idx.numpy()))
print('Classification result: {}, {}'.format(cls_result.numpy(), VOC_CLASSES[cls_result]))