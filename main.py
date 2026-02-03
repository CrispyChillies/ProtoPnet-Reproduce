import os
import sys
# Ensure local modules are imported first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
from dataset import NIHDataset  # Import NIHDatasetP
import model
import push
import prune
import train_and_test as tnt
import save
from logger_utils import create_logger
from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import data_path, csv_file, train_batch_size, test_batch_size, train_push_batch_size, test_split, val_split

# Create train/val/test splits based on patient IDs
log('Creating train/val/test splits...')
train_indices, val_indices, test_indices = NIHDataset.create_train_test_split(
    csv_file, test_size=test_split, val_size=val_split, random_state=42
)

normalize = transforms.Normalize(mean=mean, std=std)

# Training transforms with augmentation
# XProtoNet: Use RandomResizedCrop to maintain aspect ratio while adding scale augmentation
train_transform = transforms.Compose([
    # RandomResizedCrop handles both resizing and cropping with scale augmentation
    # scale=(0.75, 1.0) means crop between 75-100% of original image area
    # ratio maintains chest X-ray aspect ratio (typically around 1.0 for square-ish crops)
    transforms.RandomResizedCrop(
        size=img_size, 
        scale=(0.75, 1.0),  # Crop 75-100% of image area
        ratio=(0.9, 1.1)    # Allow slight aspect ratio variation
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    normalize,
])

# Test transforms without augmentation
test_transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
    normalize,
])

# Push transforms without normalization (needed for visualization)
push_transform = transforms.Compose([
    transforms.Resize(size=(img_size, img_size)),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = NIHDataset(
    csv_file=csv_file,
    root_dir=data_path,
    transform=train_transform,
    indices=train_indices
)

train_push_dataset = NIHDataset(
    csv_file=csv_file,
    root_dir=data_path,
    transform=push_transform,
    indices=train_indices  # Use training data for push
)

test_dataset = NIHDataset(
    csv_file=csv_file,
    root_dir=data_path,
    transform=test_transform,
    indices=test_indices
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=True)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,  
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
# Add occurrence_module if it exists
if hasattr(ppnet, 'occurrence_module'):
    joint_optimizer_specs.append(
        {'params': ppnet.occurrence_module.parameters(), 'lr': joint_optimizer_lrs['occurrence_module'], 'weight_decay': 1e-3}
    )
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.last_layer.parameters(), 'lr': warm_optimizer_lrs['last_layer']}
]
if hasattr(ppnet, 'occurrence_module'):
    warm_optimizer_specs.append(
        {'params': ppnet.occurrence_module.parameters(), 'lr': warm_optimizer_lrs['occurrence_module'], 'weight_decay': 1e-3}
    )
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

ppnet.last_layer.weight.requires_grad = True
if ppnet.last_layer.bias is not None:
    ppnet.last_layer.bias.requires_grad = True

# train the model
log('start training')
log('='*80)
log('ITERATIVE TRAINING SCHEME (XProtoNet)')
log('='*80)

import copy

# ============================================================================
# Phase 1: WARM-UP (5 epochs)
# Train only: add_on_layers, occurrence_module, prototype_vectors
# Freeze: features (backbone), last_layer
# ============================================================================
log('\n' + '='*80)
log('PHASE 1: WARM-UP TRAINING')
log('Training: add_on_layers + occurrence_module + prototypes')
log('Frozen: backbone features + last_layer')
log('='*80 + '\n')

tnt.warm_only(model=ppnet_multi, log=log)
for epoch in range(num_warm_epochs):
    log('Warm-up epoch: \t{0}/{1}'.format(epoch, num_warm_epochs))
    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                  class_specific=class_specific, coefs=coefs, log=log)
    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'warmup', accu=accu,
                                target_accu=0.30, log=log)

# ============================================================================
# Phase 2: ITERATIVE TRAINING
# Cycle: Joint Training → Push → Last Layer Training → (Optional Pruning)
# ============================================================================
log('\n' + '='*80)
log('PHASE 2: ITERATIVE JOINT TRAINING')
log('Training: All layers (features + add_on + occurrence + prototypes + last_layer)')
log('='*80 + '\n')

# Switch to joint training mode (unfreeze backbone)
tnt.joint(model=ppnet_multi, log=log)

for epoch in range(num_warm_epochs, num_train_epochs):
    log('\n' + '-'*80)
    log('Epoch: \t{0}/{1}'.format(epoch, num_train_epochs))
    log('-'*80)
    
    # ========================================================================
    # Step 1: JOINT TRAINING
    # Train all layers together
    # ========================================================================
    log('Step 1: Joint training (features + prototypes + last_layer)...')
    joint_lr_scheduler.step()
    _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                  class_specific=class_specific, coefs=coefs, log=log)
    
    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.50, log=log)
    
    # ========================================================================
    # Step 2: PUSH PROTOTYPES (on scheduled epochs)
    # Replace prototype vectors with nearest training patches
    # ========================================================================
    if epoch >= push_start and epoch in push_epochs:
        log('\n' + '>'*60)
        log('PUSH EPOCH: Replacing prototypes with nearest patches')
        log('>'*60)
        
        push.push_prototypes(
            train_push_loader,
            prototype_network_parallel=ppnet_multi,
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function,
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,
            epoch_number=epoch,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.50, log=log)
        
        # ====================================================================
        # Step 3: LAST LAYER TRAINING
        # Fine-tune classification weights after push
        # Allow weights to diverge from initial value of 1.0
        # ====================================================================
        log('\n' + '>'*60)
        log('LAST LAYER TRAINING: Fine-tuning classification weights')
        log('>'*60)
        
        tnt.last_only(model=ppnet_multi, log=log)
        for i in range(20):
            log('Last layer iteration: \t{0}/20'.format(i))
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
            accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, 
                                       model_name=str(epoch) + '_' + str(i) + 'push', 
                                       accu=accu, target_accu=0.50, log=log)
        
        # ====================================================================
        # Step 4: PRUNING (Optional)
        # Remove prototypes with negative weights in last_layer
        # Uncomment to enable pruning
        # ====================================================================
        # log('\n' + '>'*60)
        # log('PRUNING: Removing prototypes with negative weights')
        # log('>'*60)
        # prune.prune_prototypes(ppnet_multi, 
        #                        prune_threshold=-0.5,
        #                        preprocess_input_function=preprocess_input_function,
        #                        log=log)
        
        # ====================================================================
        # Return to joint training mode for next iteration
        # ====================================================================
        log('\n' + '>'*60)
        log('Returning to joint training mode')
        log('>'*60)
        tnt.joint(model=ppnet_multi, log=log)

log('\n' + '='*80)
log('TRAINING COMPLETE')
log('='*80)
   
logclose()

