import torch
import utils as ut
import model_fn as mf
from torch import nn, optim
from torch.optim import lr_scheduler

def main():
    """ A function that calls and runs the other functions. """
    
    # Calls user_args() and prints example command line arguments
    args = ut.user_arg()
    print("Command Line Arguments:\n- data directory =", args.data_dir, "\n- architect =", args.arch, "\n- gpu =", args.gpu, "\n  ...  ")
    
    # Calls the data_loader() function
    trainloader, validateloader, testloader, classes = ut.data_loader(args)
    
    # Calls build_model() and prints out the classifier
    model = mf.build_model(args, classes)
    if args.arch == 'resnet18':
        print(model.fc)
    else:
        print(model.classifier) 
    
    # Defines loss function, learning rate, scheduler and optimizer
    criterion = nn.NLLLoss()
    learn_rate = args.learn_rate
    if args.arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr = learn_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)
    
    # Sets epochs to value provided by user_arg()
    epochs = args.epochs
    
    # Sets device to cuda if available
    if args.gpu == True and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif args.gpu == True and not torch.cuda.is_available():
        args.gpu = False
        print("GPU is not available, running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        
    # Calls the train_model() function
    mf.train_model(model, trainloader, validateloader, criterion, optimizer, scheduler, device, epochs)
    
    # Call save_checkpoint() function
    mf.save_checkpoint(args, model, epochs, optimizer)
    
# Calls main() function
if __name__ == "__main__":
    main() 
