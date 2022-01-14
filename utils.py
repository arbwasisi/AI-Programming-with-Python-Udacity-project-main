import argparse
import torch
import numpy as np
from torchvision import transforms, models, datasets
from PIL import Image

def user_arg():
    """ Function that parses arguments and creates user friendly prompts. """
    
    # Creates argument parser object
    parser = argparse.ArgumentParser()
    
    # Adds command line arguments to parser object
    parser.add_argument('--data_dir', help = 'Define path to the Image set', default = 'flowers')
    parser.add_argument('--arch', help = 'Select model architect', choices = 
                       ['densenet121', 'resnet18', 'vgg16'], default = 'densenet121')
    parser.add_argument('--hidden_units', type = int, help = 'Set number of hidden units', default = 1000)
    parser.add_argument('--learn_rate', type = float, help = 'Set learning rate', default = 0.001)
    parser.add_argument('--epochs', type = int, help = 'Set number of epochs', default = 10)
    parser.add_argument('--image', help = 'Define path to an image', default = '/test/98/image_07777.jpg')
    parser.add_argument('--topk', type = int, help = 'Return top K most likely classes', default = 5)
    parser.add_argument('--json_file', help = 'Load a JSON file', default = 'cat_to_name.json')
    parser.add_argument('--gpu', type = bool, help = 'Run on GPU if available(T/F)', default = False)
    
    # Returns parsed arguments
    args_parsed = parser.parse_args()
    return args_parsed

def data_loader(args):
    """ A function that loads and transforms the Image set. """
    
    # Defines data directory
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defines data transforms
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    general_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])
    
    # Loads the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform = general_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = general_transforms)
    
    # Defines the dataloaders, using the image datasets
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validateloader = torch.utils.data.DataLoader(validate_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = True)
    
    classes = train_data.class_to_idx
    
    return trainloader, validateloader, testloader, classes

def process_image(image):
    """ A function that process the image for inference."""

    # Opens image and defines weight and height variable
    image = Image.open(image)
    width, height = image.size
    
    # Resize image with thumbnail
    if width > height:
        image.thumbnail((1000, 256))
    else:
        image.thumbnail((256, 1000))
    
    # Sets margins and crops images
    left = (image.size[0]-224)/2
    lower = (image.size[1]-224)/2
    right = left + 224
    upper = lower + 224
    
    image = image.crop((left, lower, right, upper))
    
    # Normalize and transpose image
    np_img = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    image = (np_img - mean)/std
    image = image.transpose((2, 0, 1))
    
    return image

def predict(image_path, model, topk, json_file, device):
    """ Predicts the topK classes for a given image."""
    
    # Set model to evaluation mode and available device(gpu/cpu)
    model.eval()
    model.to(device)
    
    # Predicts the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    
    # Gets the max probabilty of an image 
    output = model.forward(image)
    ps = torch.exp(output)
    top_ps, labels = ps.topk(topk)
    top_ps = top_ps.detach().numpy().tolist()[0] 
    labels = labels.detach().numpy().tolist()[0]
    
    # Gets the labels from class_to_idx 
    inverted_index = {value: key for key, value in model.class_to_idx.items()}
    flower_labs = [json_file[inverted_index[label]] for label in labels]   
    
    return top_ps, flower_labs
    
    