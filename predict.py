import utils as ut
import model_fn as mf
import torch
import json

def main():
    """ Main function to call and runs the other functions. """
    
    # Calls user_args() and prints example command line arguments
    args = ut.user_arg()
    print("Command Line Arguments:\n- image directory =", args.image, "\n- topk =", args.topk, "\n- category file =", args.json_file, "\n  ...  ")
    
    # Calls the load function
    model = mf.load_checkpoint('checkpoint.pth', args)

    # Sets device to cuda if available
    if args.gpu == True and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif args.gpu == True and not torch.cuda.is_available():
        args.gpu = False
        print("GPU is not available, running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    # Loads json file with 
    with open(args.json_file, 'r') as fl:
        cat_file = json.load(fl)
    
    # Defines image and topk from user_args() function
    image = args.data_dir + args.image
    topk = args.topk
    
    # Calles the predict() function and prints results
    probability, labels = ut.predict(image, model, topk, cat_file, device)
    print("Results:")      
    for name, prob in zip(labels, probability):
        print('{}: '.format(name) + '%d%%'%(100 * prob))
    
    # Prints name of flower
    name = cat_file[image.split('/')[2]]
    print("Name of Flower: {}".format(name))
    
    # Prints message if prediction fails or pass
    if name == labels[0]:
        print("Prediction Succesfull!")
    else:
        print("Prediction Failed :(" + "\nGive it one more shot!")
    
# Calls main() function
if __name__ == "__main__":
    main()    
    