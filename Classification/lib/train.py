import numpy as np
import torch
import time
import copy

from tqdm import tqdm

def train_model(device, model, dataloaders, dataset_sizes,
                criterion, optimizer, scheduler=None,
                metric=None, num_epochs=25, load_best_weights=True,
                dtype_input=torch.float, dtype_label=torch.long):
    since = time.time()
    
    history_loss = {'train': [], 'val': []}
    history_metric = {'train': [], 'val': []}

    # Find the initial score  
    print("Find the best initial score ...")    
    best_model_wts = copy.deepcopy(model.state_dict())
    model.eval()
    running_metric = 0.0
    
    # Iterate over data.
    for ii, (inputs, labels) in tqdm(enumerate(dataloaders['val']), total=len(dataloaders['val'])):
        inputs = inputs.to(device, dtype=dtype_input)
        labels = labels.to(device, dtype=dtype_label)

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # statistics
        running_metric += metric(labels, outputs) * inputs.size(0)
        
    best_metric = running_metric / dataset_sizes['val']

    # Training loop
    # Iterate over num_epochs
    print("Start training ...")
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_metric = 0.0

            # Iterate over data.
            for ii, (inputs, labels) in tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase])):
                inputs = inputs.to(device, dtype=dtype_input)
                labels = labels.to(device, dtype=dtype_label)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_metric += metric(labels, outputs) * inputs.size(0)
                    
            if phase == 'train':
                if scheduler: scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_metric = running_metric / dataset_sizes[phase]
            
            history_loss[phase].append(epoch_loss)
            history_metric[phase].append(epoch_metric)
           
            # deep copy the model
            if (metric is not None) and (phase == 'val'):
                
                if (metric.target.lower() == 'max') and (epoch_metric > best_metric):
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                elif (metric.target.lower() == 'min') and (epoch_metric < best_metric):
                    best_metric = epoch_metric
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
        print('Epoch {}/{} - loss: {:.4f} - val_loss: {:.4f} - {}: {:.4f} - val_{}: {:.4f}'.format(
            epoch + 1, num_epochs,
            history_loss['train'][-1], history_loss['val'][-1],
            metric.name, history_metric['train'][-1],
            metric.name, history_metric['val'][-1]))

    time_elapsed = time.time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val_{}: {:4f}'.format(metric.name, best_metric))

    # load best model weights
    if load_best_weights: model.load_state_dict(best_model_wts)
    
    return {'history_loss': history_loss, 'history_metric': history_metric}
