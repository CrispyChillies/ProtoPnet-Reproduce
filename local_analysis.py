##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from logger_utils import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse

# Define NIH ChestXray disease labels (14 diseases)
NIH_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
    'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-model', nargs=1, type=str, required=True, help='Path to the .pth model file')
parser.add_argument('-img', nargs=1, type=str, required=True, help='Path to the input X-ray image')
parser.add_argument('-threshold', nargs=1, type=float, default=[0.5], help='Threshold for disease prediction (default: 0.5)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test image to be analyzed
test_image_path = args.img[0]
prediction_threshold = args.threshold[0]

# load the model
load_model_path = args.model[0]
model_name = os.path.basename(load_model_path)
image_name = os.path.basename(test_image_path).split('.')[0]

# Save to /kaggle/working instead of read-only input directory
save_analysis_path = os.path.join('/kaggle/working', 'xprotonet_analysis', image_name)
makedir(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

log('load model from ' + load_model_path)
log('test image: ' + test_image_path)
log('prediction threshold: ' + str(prediction_threshold))

ppnet = torch.load(load_model_path, weights_only=False)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
ppnet_multi.eval()

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
num_classes = ppnet_multi.module.num_classes

log('img_size: ' + str(img_size))
log('prototype_shape: ' + str(prototype_shape))
log('num_classes: ' + str(num_classes))

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# Removed test accuracy check section for simplicity


##### SANITY CHECK
# Note: For XProtoNet with NIH dataset, we don't have prototype image files
# from training, so we skip the sanity check section
log('Skipping prototype sanity checks (not applicable for current setup)')


##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

img_pil = Image.open(test_image_path).convert('RGB')
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

images_test = img_variable.cuda()

# Forward pass - XProtoNet returns 3 values: logits, min_distances, occurrence_maps
log('Running forward pass...')
with torch.no_grad():
    logits, min_distances, occurrence_maps = ppnet_multi(images_test)

conv_output, distances = ppnet.push_forward(images_test)
prototype_activations = ppnet.distance_2_similarity(min_distances)
prototype_activation_patterns = ppnet.distance_2_similarity(distances)
if ppnet.prototype_activation_function == 'linear':
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

# Multi-label prediction using sigmoid
probabilities = torch.sigmoid(logits).cpu().numpy()[0]
predictions = probabilities > prediction_threshold

log('\n' + '='*60)
log('PREDICTION RESULTS (Multi-label)')
log('='*60)
for idx, label in enumerate(NIH_LABELS):
    status = "POSITIVE" if predictions[idx] else "negative"
    log('{:<25}: {:.4f}  [{}]'.format(label, probabilities[idx], status))
log('='*60 + '\n')

idx = 0
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test, idx)

##### VISUALIZE OCCURRENCE MAPS FOR TOP PREDICTED DISEASES
# This is the key feature of XProtoNet - direct visualization of disease regions
makedir(os.path.join(save_analysis_path, 'occurrence_maps'))

log('='*60)
log('VISUALIZING OCCURRENCE MAPS (XProtoNet Feature)')
log('='*60)

# Get top diseases by probability
top_k = min(5, num_classes)
top_k_indices = probabilities.argsort()[-top_k:][::-1]

# Get occurrence maps from model output
occ_maps = occurrence_maps.cpu().numpy()[0]  # Shape: (Num_Prototypes, H, W)

# Get prototype class identity
proto_class_identity = ppnet.prototype_class_identity.cpu().numpy()

for rank, disease_idx in enumerate(top_k_indices):
    disease_name = NIH_LABELS[disease_idx]
    prob = probabilities[disease_idx]
    
    log(f'Top {rank+1}: {disease_name} (prob={prob:.4f})')
    
    # Find prototypes for this disease
    prototypes_for_disease = np.where(proto_class_identity[:, disease_idx] == 1)[0]
    log(f'  Number of prototypes for this disease: {len(prototypes_for_disease)}')
    
    # Aggregate occurrence maps for all prototypes of this disease
    disease_heatmap = np.zeros((occ_maps.shape[1], occ_maps.shape[2]))
    
    for proto_idx in prototypes_for_disease:
        disease_heatmap += occ_maps[proto_idx]
    
    # Normalize heatmap
    disease_heatmap = np.maximum(disease_heatmap, 0)
    if np.max(disease_heatmap) > 0:
        disease_heatmap /= np.max(disease_heatmap)
    
    # Upsample to image size
    upsampled_heatmap = cv2.resize(disease_heatmap, (img_size, img_size), 
                                   interpolation=cv2.INTER_CUBIC)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * upsampled_heatmap), cv2.COLORMAP_JET)
    heatmap_colored = np.float32(heatmap_colored) / 255
    heatmap_colored = heatmap_colored[..., ::-1]  # BGR to RGB
    
    # Overlay on original image
    overlayed_img = 0.5 * original_img + 0.4 * heatmap_colored
    
    # Save visualizations
    save_name = f'top{rank+1}_{disease_name}_prob{prob:.2f}_occurrence_map.png'
    plt.imsave(os.path.join(save_analysis_path, 'occurrence_maps', save_name), overlayed_img)
    
    # Also save just the heatmap
    plt.imsave(os.path.join(save_analysis_path, 'occurrence_maps', 
                           f'top{rank+1}_{disease_name}_heatmap_only.png'), upsampled_heatmap, cmap='jet')
    
    log(f'  Saved: {save_name}')

log('='*60 + '\n')

##### PROTOTYPE ACTIVATION PATTERNS (Additional Analysis)
# Show which prototypes are most activated for this specific image
makedir(os.path.join(save_analysis_path, 'prototype_activations'))

log('='*60)
log('TOP 10 MOST ACTIVATED PROTOTYPES')
log('='*60)

array_act, sorted_indices_act = torch.sort(prototype_activations[idx])
for i in range(1, 11):
    proto_idx = sorted_indices_act[-i].item()
    activation_val = array_act[-i].item()
    
    # Find which disease(s) this prototype belongs to
    proto_diseases = np.where(proto_class_identity[proto_idx] == 1)[0]
    disease_names = [NIH_LABELS[d] for d in proto_diseases]
    
    log(f'Rank {i}: Prototype {proto_idx}')
    log(f'  Activation: {activation_val:.4f}')
    log(f'  Associated diseases: {", ".join(disease_names)}')
    
    # Visualize activation pattern
    activation_pattern = prototype_activation_patterns[idx][proto_idx].detach().cpu().numpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                              interpolation=cv2.INTER_CUBIC)
    
    # Normalize and apply colormap
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    if np.amax(rescaled_activation_pattern) > 0:
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
    
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    
    # Save
    disease_str = "_".join(disease_names[:2])  # Limit to 2 diseases for filename
    save_name = f'rank{i}_proto{proto_idx}_{disease_str}_act{activation_val:.2f}.png'
    plt.imsave(os.path.join(save_analysis_path, 'prototype_activations', save_name), overlayed_img)
    log(f'  Saved: {save_name}')
    log('--------------------------------------------------------------')

log('='*60 + '\n')

##### SUMMARY
log('='*60)
log('ANALYSIS COMPLETE')
log('='*60)
log(f'Results saved to: {save_analysis_path}')
log('Files generated:')
log('  - original_img.png: Input image (preprocessed)')
log('  - occurrence_maps/: XProtoNet disease localization heatmaps')
log('  - prototype_activations/: Individual prototype activation patterns')

# Count positive predictions
num_positive = np.sum(predictions)
log(f'\nTotal diseases detected (>{prediction_threshold}): {num_positive}/{num_classes}')
if num_positive > 0:
    positive_diseases = [NIH_LABELS[i] for i in range(num_classes) if predictions[i]]
    log(f'Detected diseases: {", ".join(positive_diseases)}')
else:
    log('No diseases detected above threshold.')

log('='*60)

logclose()


