from helpers import ParameterStorage
from learner import Learner
from data_loader import load_cifar10
from networks import CNN, FNN
import torch
import datetime
import os

if __name__ == '__main__':
    # Set the parameters and initialize the model
    learning_rate = 0.001
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = CNN().to(device)
    model_name = 'CNN' if isinstance(classifier, CNN) else 'FNN'
    epochs = 5

    # Create a folder with the current timestamp to save the results
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_folder = os.path.join('00_Results', now)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Load the CIFAR10 dataset using the load_cifar10 function from data_loader.py
    train_loader, test_loader = load_cifar10(batch_size)

    # Set the optimizer and its parameters based on the optimizer choice (SGD or Adam)
    optimizer = "Adam"
    optimizer_params = {'momentum': 0.9} if optimizer == "SGD" else {
        'weight_decay': 0.0001}

    # Initialize the Learner object with the parameters, including the optimizer parameters using **kwargs
    my_learner = Learner(learning_rate=learning_rate,
                         optimizer=optimizer,
                         model=classifier,
                         device=device,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         epochs=epochs,
                         result_folder=result_folder,
                         model_name=model_name,
                         **optimizer_params)

    # Create a ParameterStorage object and store the current parameters used in the model
    parameter_storage = ParameterStorage(result_folder, my_learner)
    parameter_storage.store()

    # Train the model using the fit function from the Learner object
    my_learner.fit()
