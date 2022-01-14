import torch
import copy
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
from collections import OrderedDict

def build_model(args, classes):
      """ A function that loads and builds the model. """
      
      # Load pre-trained model
      selected_model = args.arch
      model = getattr(models, selected_model)(pretrained=True)

      # Freeze model parameters for network features
      for param in model.parameters():
          param.requires_grad = False
      
      # Sets values for input layer, hidden units, and output layer  
      if selected_model == 'resnet18':
          input_layer = model.fc.in_features
      elif selected_model == 'vgg16':
          input_layer = model.classifier[0].in_features
      else:
          input_layer = model.classifier.in_features
      hidden_units = args.hidden_units
      output_layer = 102
      
      # Build new classifier and replace current
      classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(p=0.5)),
        ('fc1', nn.Linear(input_layer, hidden_units)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, output_layer)),
        ('output', nn.LogSoftmax(dim=1))
      ]))
      
      if selected_model == 'resnet18':
          model.fc = classifier
      else:
          model.classifier = classifier
      model.class_to_idx = classes
    
      return model 
    
def validation(criterion, dataloader, model, device):
    """ A function that trains the model on a validation set. """
    
    # Removes the dropout function for inference
    model.eval()
    val_loss = 0
    accuracy = 0
    
    # Use for loop to pass images into network
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
       
        # Calculate loss and accuracy of network
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        ps = torch.exp(output).data
        equality = (labels.data == ps.max(dim = 1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
     
    val_loss /= len(dataloader)
    accuracy /= len(dataloader)
    
    return val_loss, accuracy

def train_model(model, trainloader, validateloader, criterion, optimizer, scheduler, device, epochs):
    """ A function that trains the model and prints out the results. """
 
    # Set number of iteration
    print_every = 40
    step = 0
   
    # Set to available device
    model.to(device)
    optim_model = copy.deepcopy(model.state_dict())
    best_acc = 0

    # Loops through epochs
    for e in range(epochs):
        model.train()
        scheduler.step()
        running_loss = 0
    
        # Iterates through data
        for images, labels in trainloader:
            step += 1
        
            # Sets data to device (gpu/cpu)
            images = images.to(device)
            labels = labels.to(device)
        
            # Removes gradient descent for each loops
            optimizer.zero_grad()
        
            # Passes data through network, calculate loss and update weights
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            # Define validation loss and accuracy with validate function
            if step % print_every == 0:
            
                # Turn off gradient for validation
                with torch.no_grad():
                    val_loss, accuracy = validation(criterion, validateloader, model, device)
            
                # Print model results
                print("Epoch: {}/{} ".format(e+1, epochs),
                    "Training Loss: {:.3f} ".format(running_loss/print_every),
                    "Validation Loss: {:.3f} ".format(val_loss),
                    "Validation Accuracy: {:.3f}".format(accuracy))
            
                # Copy most optim model
                if accuracy > best_acc:
                    best_acc = accuracy
                    optim_model = copy.deepcopy(model.state_dict())
            
                running_loss = 0
            
    print('Best Accuracy: {:.3f}'.format(best_acc))
    print(' ~ End of Training')

    # Load most optimal model
    model.load_state_dict(optim_model)

def save_checkpoint(args, model, epochs, optimizer):
    """ A function that saves the model attributes to a checkpoint."""
    
    # Save model, hyperparameters, and state_dict to checkpoint
    checkpoint = {
        'model': args.arch,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_index': model.class_to_idx
     }
    if args.arch == 'resnet18':
        checkpoint['classifier'] = model.fc
    else:
        checkpoint['classifier'] = model.classifier
        
    torch.save(checkpoint, 'checkpoint.pth')
    
def load_checkpoint(filepath):
    """ A function that loads and rebuilds the model. """
    
    # Loads check point
    checkpoint = torch.load(filepath)
    
    # Loads and rebuilds the model 
    model = getattr(models, checkpoint['model'])(pretrained=True)
    # Freeze model parameters 
    for param in model.parameters():
        param.requires_grad = False
    
    # Loads model classifier and optimizer from checkpoint
    model.optimizer = checkpoint['optimizer']
    if checkpoint['model'] == 'resnet18':
        model.fc = checkpoint['classifier']
    else:
        model.classifier = checkpoint['classifier']
    # Loads weight, baises and the mapping class
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_index']
    
    return model 
    
    