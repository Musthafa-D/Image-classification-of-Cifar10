# Importing the necessary libraries and modules
import os

# Stores the parameters used for training the model in a text file.
class ParameterStorage:
    def __init__(self, result_folder, learner):
        self.folder = os.path.join(result_folder, "Parameters.txt")
        self.learner = learner

    def store(self):
        with open(self.folder, 'w') as f:
            f.write(
                'These are the parameters that were used\n\n')
            f.write(f'Learning Rate: {self.learner.learning_rate}\n')
            f.write(f'Model name: {self.learner.model_name}\n')
            f.write(f'Model:- \n {self.learner.model}\n')
            # f.write(
            #     f'Optimizer: {self.learner.optimizer.__class__.__name__}\n')
            f.write(
                f'Optimizer: {self.learner.optimizer}\n')
            f.write(f'Device: {self.learner.device}\n')
            f.write(f'Number of epochs: {self.learner.epochs}\n')


class FigureStorage:  # Saves the figures generated during training and evaluation in specified formats
    def __init__(self, result_folder, types=('png', 'pdf')):
        self.images_folder = os.path.join(result_folder, 'Images')
        self.types = types
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)

    def store(self, fig, file_name):
        if "png" in self.types:
            self.save_as_png(fig, file_name)
        if "pdf" in self.types:
            self.save_as_pdf(fig, file_name)

    def save_as_png(self, fig, file_name):
        file_path = os.path.join(self.images_folder, file_name + ".png")
        fig.savefig(file_path)

    def save_as_pdf(self, fig, file_name):
        file_path = os.path.join(self.images_folder, file_name + ".pdf")
        fig.savefig(file_path)


class DataStorage:  # Stores train and test accuracies and losses during the training process
    def __init__(self):
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_losses = []
        self.test_losses = []

    def store(self, vals: list):
        self.train_losses.append(vals[0])
        self.train_accuracies.append(vals[1])
        self.test_losses.append(vals[2])
        self.test_accuracies.append(vals[3])
