# AI-Programming-with-Python-Udacity-project
Final Project from Udacity's AI Programming course.

In this project, we are tasked with building an image classifier to classify a dataset of image flowers.
Initaily we build and train a neural network on Jupyter Notebook, then we build a command line application.

The project is broken down into multiple steps:

- Load and preprocess the image dataset
- Train the image classifier on the dataset
- Use the trained classifier to predict image content

The repo is mainly composed of modules that run on the commandline. 

In `utils.py`, I create the command line input arguments and instructions for the users to follow
and run the app. `model_fn.py` downloads a pretrained neural network with pytorch, trains the model, and saves
it to a checkpoint which we can load from for later use. That last two modules `train.py` and `predict.py` are
`main()` calls that take the users inputs, builds and trains the module and provides a prediction, for the 
given image, the likely top 5 flowers in the image.
