import torch
import numpy as np
import math
import cv2

def sigmoid(x):
 return 1/(1 + np.exp(-x))

VOC_CLASSES = (  # always index 0
'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus', 'car', 'cat', 'chair',
'cow', 'diningtable', 'dog', 'horse',
'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor')

nn_out = np.load('dog52.npy')
print('NN output shape: ', nn_out.shape)
pred = torch.from_numpy(nn_out)
B, abC, H, W = pred.size()
nn_out_flat = torch.flatten(pred)


KA = 5
NC = 20
prediction = pred.permute(0, 2, 3, 1).contiguous().view(B, -1, abC)
#print(prediction.size()) # 169 x 125

conf_pred = prediction[..., :KA].contiguous().view(B, -1, 1)
cls_pred = prediction[..., 1*KA : (1+NC)*KA].contiguous().view(B, -1, NC)
txtytwth_pred = prediction[..., (1+NC)*KA:].contiguous().view(B, -1, 4)

conf_pred = conf_pred[0]            #[H*W*KA, 1]
cls_pred = cls_pred[0]              #[H*W*KA, NC]
txtytwth_pred = txtytwth_pred[0]    #[H*W*KA, 4]

max_conf, max_conf_idx = torch.max(conf_pred, dim=0)
max_conf_cls = cls_pred[max_conf_idx]
max_cls, cls_result = torch.max(max_conf_cls, dim=1)

print('Max_Confidence: {}'.format(max_conf.numpy()))
print('Max_Confidence index: {}'.format(max_conf_idx.numpy()))
print('Classification result: {}, {}'.format(cls_result.numpy(), VOC_CLASSES[cls_result]))

# calculate original max confidence index
d3 = math.floor(max_conf_idx/(W*KA))
d2 = math.floor((max_conf_idx - d3*W*KA)/KA)
d1 = max_conf_idx.item() - d3*W*KA - d2*KA
print('conf idx ', d1, d2, d3)
assert max_conf.item() == nn_out[0,d1,d2,d3]

# calculate original max conf class index
d4 = KA + d1*NC + cls_result.item()
print('class idx ',d4, d2, d3)
assert max_cls.item() == nn_out[0,d4,d2,d3]

# load image
im = cv2.imread('dog52.jpg')  # BGR
#iH, iW, iC = im.shape
#print('image shape: ', iH, iW, iC)
im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
im = np.ascontiguousarray(im)  # contiguous
im = torch.from_numpy(im)
iC, iH, iW = im.size()
print('NN input shape: ',iC, iH, iW)
im = im.float()  # uint8 to float
im /= 255  # 0 - 255 to 0.0 - 1.0
imf=torch.flatten(im)

grid_size = H
nclasses = NC
boxes = KA
epsilon = 1.0/255

output_channels = boxes * (nclasses + 5)

f = open("dog52.vnnlib", "w")

# Declear inputs
f.write("; input "+str(iC)+" x "+str(iH)+" x "+str(iW)+" brightness epsilon "+str(epsilon)+"\n\n")
for i in range(imf.size(dim=0)):
   f.write("(declare-const X_" + str(i) + " Real)\n")
   
# Declear outputs
f.write("\n; output " + str(output_channels) + " x " + str(grid_size) + " x " + str(grid_size) + "\n\n")
for i in range(nn_out_flat.size(dim=0)):
   f.write("(declare-const Y_" + str(i) + " Real)\n")

# Declear input constraints
f.write("\n; input constraints\n\n")
for i in range(imf.size(dim=0)):
   val = imf[i].item()
   ub = min(val+epsilon, 1.0)
   lb = max(val-epsilon, 0.0)
   f.write("(assert (<= X_" + str(i) + " " + str(ub) + "))\n")
   f.write("(assert (>= X_" + str(i) + " " + str(lb) + "))\n")

# Declear output constraints
f.write("\n; output constraints\n\n")

c_conf = d1
c_class = d4
y = d3
x = d2
geo = 4 # tx, ty, tw, th
max_conf_idx = c_conf*grid_size*grid_size + y*grid_size + x
max_class_idx = c_class*grid_size*grid_size + y*grid_size + x

#print(nn_out_flat[max_conf_idx])
val = nn_out_flat[max_conf_idx].item()
#conf = sigmoid(val)
#print('Confidence: ', conf)
#print(nn_out_flat[max_class_idx])

# confidence constraints
for idx in range(boxes):
  if idx != c_conf:
      conf_idx = idx*grid_size*grid_size + y*grid_size + x
      f.write("(assert (>= Y_" + str(max_conf_idx) + " " + "Y_" + str(conf_idx) + "))\n")

# classification constraints
for idx in range(boxes+c_conf*nclasses, boxes+(c_conf+1)*nclasses):
  if idx != c_class:
      class_idx = idx*grid_size*grid_size + y*grid_size + x
      f.write("(assert (>= Y_" + str(max_class_idx) + " " + "Y_" + str(class_idx) + "))\n")

# bounding box constraints
box_epsilon = 0.01
for idx in range(boxes+boxes*nclasses+c_conf*geo, boxes+boxes*nclasses+(c_conf+1)*geo):
         geo_idx = idx*grid_size*grid_size + y*grid_size + x
         val = nn_out_flat[geo_idx].item()
         ub = val + box_epsilon
         lb = val - box_epsilon
         f.write("(assert (<= Y_" + str(geo_idx) + " " + str(ub) + "))\n")
         f.write("(assert (>= Y_" + str(geo_idx) + " " + str(lb) + "))\n")

f.close()
