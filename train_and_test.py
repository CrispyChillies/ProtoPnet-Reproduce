import time
import torch

from helpers import list_of_distances, make_one_hot

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_occurrence_loss = 0
    
    # Multi-label classification loss
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()  # target is now multi-label: (batch_size, num_classes)

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances, occurrence_map = model(input)

            # compute loss - Multi-label classification
            cross_entropy = bce_loss(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # Multi-label: target shape is (batch_size, num_classes)
                # prototype_class_identity shape is (num_prototypes, num_classes)
                # We need to compute masks based on which diseases are present
                
                # For each sample, get which prototypes correspond to positive labels
                # prototypes_of_positive_class: (batch_size, num_prototypes)
                prototypes_of_positive_class = torch.matmul(target, model.module.prototype_class_identity.t().cuda())
                prototypes_of_positive_class = (prototypes_of_positive_class > 0).float()
                
                # Cluster cost: pull features close to prototypes of diseases present in image
                if prototypes_of_positive_class.sum() > 0:
                    inverted_distances = (max_dist - min_distances) * prototypes_of_positive_class
                    cluster_cost = torch.sum(max_dist - inverted_distances) / (prototypes_of_positive_class.sum() + 1e-5)
                else:
                    cluster_cost = torch.tensor(0.0).cuda()

                # Separation cost: push features away from prototypes of diseases NOT present
                prototypes_of_negative_class = torch.matmul(1 - target, model.module.prototype_class_identity.t().cuda())
                prototypes_of_negative_class = (prototypes_of_negative_class > 0).float()
                
                if prototypes_of_negative_class.sum() > 0:
                    # We want to maximize distance (minimize inverted distance) for wrong prototypes
                    inverted_distances_to_negative = (max_dist - min_distances) * prototypes_of_negative_class
                    separation_cost = torch.sum(inverted_distances_to_negative) / (prototypes_of_negative_class.sum() + 1e-5)
                else:
                    separation_cost = torch.tensor(0.0).cuda()

                # Average separation cost
                if prototypes_of_negative_class.sum(dim=1).min() > 0:
                    avg_separation_cost = torch.sum(min_distances * prototypes_of_negative_class, dim=1) / \
                                         (torch.sum(prototypes_of_negative_class, dim=1) + 1e-5)
                    avg_separation_cost = torch.mean(avg_separation_cost)
                else:
                    avg_separation_cost = torch.tensor(0.0).cuda()
                
                # Occurrence Loss: L1 regularization to encourage sparse occurrence maps
                # This encourages the model to focus on specific regions
                occurrence_loss = torch.tensor(0.0).cuda()
                if hasattr(model.module, 'occurrence_module'):
                    # Get occurrence maps from the model's intermediate outputs
                    # We need to recompute or store during forward pass
                    # For now, we'll add L1 on the occurrence module weights
                    occurrence_loss += torch.mean(torch.sum(torch.abs(occurrence_map), dim=(1, 2, 3)))
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)
                occurrence_loss = torch.tensor(0.0).cuda()

            # evaluation statistics - Multi-label accuracy
            # Use threshold of 0.5 for multi-label prediction
            predicted = (torch.sigmoid(output.data) > 0.5).float()
            n_examples += target.size(0)

            # Exact match accuracy (all labels must match)
            n_correct += (predicted == target).sum().item() 
            n_examples += target.numel() # Tổng số phần tử (batch_size * num_classes)

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item() if isinstance(separation_cost, torch.Tensor) else separation_cost
            total_avg_separation_cost += avg_separation_cost.item() if isinstance(avg_separation_cost, torch.Tensor) else avg_separation_cost
            total_occurrence_loss += occurrence_loss.item() if isinstance(occurrence_loss, torch.Tensor) else occurrence_loss

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          + coefs.get('occur', 0.01) * occurrence_loss)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 0.08 * separation_cost + 1e-4 * l1 + 0.01 * occurrence_loss
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1
                          + coefs.get('occur', 0.01) * occurrence_loss)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1 + 0.01 * occurrence_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\toccurrence: \t{0}'.format(total_occurrence_loss / n_batches))
    log('\taccu (exact): \t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)


def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    if hasattr(model.module, 'occurrence_module'):
        for p in model.module.occurrence_module.parameters():
            p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    if hasattr(model.module, 'occurrence_module'):
        for p in model.module.occurrence_module.parameters():
            p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    if hasattr(model.module, 'occurrence_module'):
        for p in model.module.occurrence_module.parameters():
            p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
