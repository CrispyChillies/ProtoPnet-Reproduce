base_architecture = 'resnet50'  # Good for medical images
img_size = 448
prototype_shape = (280, 128, 1, 1)  # 20 prototypes per class * 14 classes
num_classes = 14  # NIH ChestX-ray14 has 14 disease classes
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = 'nih_001'

# NIH ChestX-ray14 dataset paths
data_path = '/kaggle/input/data'  # Root directory containing images_001, images_002, etc.
csv_file = '/kaggle/input/data/Data_Entry_2017.csv'  # Path to the CSV file

# No separate directories needed - we'll use indices to split
train_batch_size = 16  # Smaller batch size for medical images
test_batch_size = 16
train_push_batch_size = 16

# Data split ratios
test_split = 0.15  # 15% for test
val_split = 0.1    # 10% of remaining for validation

joint_optimizer_lrs = {'features': 1e-5,      # Lower LR for pretrained features
                       'add_on_layers': 3e-4,
                       'prototype_vectors': 3e-4,
                       'occurrence_module': 3e-4}  # Add occurrence module
joint_lr_step_size = 10

warm_optimizer_lrs = {'add_on_layers': 3e-4,
                      'prototype_vectors': 3e-4,
                      'occurrence_module': 3e-4}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': 0.08,      # Positive for multi-label (push away from negative classes)
    'l1': 1e-4,
    'occur': 0.01,    # Occurrence regularization
}

num_train_epochs = 50
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]  # Push every 5 epochs
