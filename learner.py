# Importing the necessary libraries and modules
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from helpers import DataStorage, FigureStorage


# Initialization, setting up the model, loss criterion, optimizer, and other important parameters
class Learner:
    def __init__(self, learning_rate, optimizer, model, device, train_loader, test_loader, epochs, model_name, result_folder, **kwargs):
        self.device = device
        self.learning_rate = learning_rate
        self.model = model
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate, **kwargs)
        elif optimizer.lower() == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                  lr=self.learning_rate, **kwargs)
        else:
            raise AttributeError("Choose SGD or ADAM as optimiser")

        # Store additional parameters in an instance variable
        self.additional_params = kwargs

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.model_name = model_name
        self.result_folder = result_folder
        self.data_storage = DataStorage()
        self.figure_storage = FigureStorage(self.result_folder)

    # Trains the model for one epoch and returns train_accuracy and train_loss
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #self.batch += 1

        train_accuracy = 100 * correct / total
        train_loss = running_loss / (i + 1)
        return train_accuracy, train_loss

    # Test the model for one epoch and returns test_accuracy and test_loss
    def test_one_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        test_loss = running_loss / (i + 1)
        return test_accuracy, test_loss

    # Calls the plot function to visualize the training and testing accuracies and losses
    def evaluate(self):
        self.plot()

    def store(self):
        train_accuracy, train_loss = self.train_one_epoch()
        test_accuracy, test_loss = self.test_one_epoch()

        self.data_storage.store(
            [train_loss, train_accuracy, test_loss, test_accuracy])

    # Called after every epoch to print the accuracies, losses, and confusion matrix, and to save the plots
    def hook_every_epoch(self, epoch):
        self.store()
        print(f'Epoch: {epoch+1}')
        print(
            f'Train Accuracy: {self.data_storage.train_accuracies[-1]:.2f}%, Train Loss: {self.data_storage.train_losses[-1]:.4f}')
        print(
            f'Test Accuracy: {self.data_storage.test_accuracies[-1]:.2f}%, Test Loss: {self.data_storage.test_losses[-1]:.4f}')
        cm = self._get_predictions_and_labels(self.test_loader)
        self.plot_confusion_matrix(cm, epoch)
        print('-' * 50)

    # Visualizes the training and testing accuracies and losses
    def plot(self):
        
        epoch = range(1, self.epochs + 1)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epoch, self.data_storage.train_accuracies, '-o', color='blue',
                label='training accuracy')
        ax.plot(epoch, self.data_storage.test_accuracies, '-o', color='red',
                label='testing accuracy')
        ax.set_xlabel('epochs')
        ax.set_xticks(epoch)
        ax.set_ylabel('accuracy')
        ax.set_yticks(range(0, 101, 10))
        ax.legend()
        plt.title('training vs testing accuracy')
        self.figure_storage.store(
            fig, 'accuracies')
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(epoch, self.data_storage.train_losses, '-o', color='blue',
                label='train loss')
        ax.plot(epoch, self.data_storage.test_losses, '-o', color='red',
                label='test loss')
        ax.set_xlabel('epochs')
        ax.set_xticks(epoch)
        ax.set_ylabel('loss')
        ax.set_yticks(torch.arange(0, 3, 0.2))
        ax.legend()
        plt.title('train vs test loss')
        self.figure_storage.store(
            fig, 'losses')
        plt.show()

    def confusion_matrix_torch(self, y_true, y_pred, num_classes):
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    # Returns a confusion matrix for the given data_loader
    def _get_predictions_and_labels(self, data_loader):
        all_labels = torch.tensor([], dtype=torch.long)
        all_predictions = torch.tensor([], dtype=torch.long)
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions = torch.cat((all_predictions, predicted.cpu()))
                all_labels = torch.cat((all_labels, labels.cpu()))

        num_classes = len(self.train_loader.dataset.classes)
        cm = self.confusion_matrix_torch(
            all_labels, all_predictions, num_classes)
        return cm

    # Visualizes the confusion matrix for the given epoch
    def plot_confusion_matrix(self, cm, epoch):
        cm_normalized = cm / torch.sum(cm, dim=1, keepdim=True)
        cm_percentage = cm_normalized * 100
        fig, ax = plt.subplots(figsize=(12, 12))

        # Define custom colormap with specific color shades for each range of values
        cmap = plt.cm.YlGnBu
        intervals = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        bounds = torch.tensor(intervals, dtype=torch.float32)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        cax = ax.matshow(cm_percentage, cmap=cmap, norm=norm)
        cbar = fig.colorbar(cax, ticks=intervals)
        cbar.ax.set_yticklabels(intervals)

        # Iterate over the elements of the confusion matrix
        for i in range(cm_percentage.size(0)):
            for j in range(cm_percentage.size(1)):
                z = cm_percentage[i, j]
                ax.text(j, i, '{:0.1f}'.format(z.item()), ha='center',
                        va='center', color='white' if z > 50 else 'black')

        ax.set_xlabel('predicted')
        ax.set_ylabel('true')

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(range(1, 11))
        ax.xaxis.set_ticks_position('bottom')
        ax.set_yticklabels(range(1, 11))
        plt.title('confusion matrix of test data')

        self.figure_storage.store(
            fig, f'confusion_matrix_epoch_{epoch+1}')

        plt.show()

    def save(self):
        best_train_accuracy = max(self.data_storage.train_accuracies)
        best_epoch_train = self.data_storage.train_accuracies.index(best_train_accuracy)
        best_test_accuracy = max(self.data_storage.test_accuracies)
        best_epoch_test = self.data_storage.test_accuracies.index(best_test_accuracy)
        least_train_loss = min(self.data_storage.train_losses)
        least_loss_epoch_train = self.data_storage.train_losses.index(least_train_loss)
        least_test_loss = min(self.data_storage.test_losses)
        least_loss_epoch_test = self.data_storage.test_losses.index(least_test_loss)

        print(f'Best Train Accuracy: {best_train_accuracy:.2f}% (at epoch {best_epoch_train + 1})')
        print(f'Best Test Accuracy: {best_test_accuracy:.2f}% (at epoch {best_epoch_test + 1})')
        print(f'Least Train Loss: {least_train_loss:.4f} (at epoch {least_loss_epoch_train + 1})')
        print(f'Least Test Loss: {least_test_loss:.4f} (at epoch {least_loss_epoch_test + 1})')


    def fit(self):
        """
        trains and evaluates the model for the specified number of epochs. 
        Calls the hook_every_epoch function after each epoch and evaluates the model after all epochs.
        """
        self.batch = 0
        for epoch in range(self.epochs):
            self.test_one_epoch()
            self.train_one_epoch()
            self.hook_every_epoch(epoch)
        self.test_one_epoch()
        self.evaluate()
        self.save()

    # work with this fit
    # def fit(self):
    #     self.batch = 0
    #     for epoch in range(self.epochs):
    #         self.test()
    #         self.train()
    #         self.hook()
    #         self.plot(iterative=True)
    #     self.test()
    #     self.evaluate()
    #     self.save()
