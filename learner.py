import torch
import ccbdl
import os
from plots import Accuracy_plot, Attribution_plots, Learning_rate_plot, Tsne_plot, Softmax_plot, TimePlot, Loss_plot, Precision_plot, Recall_plot, F1_plot, Hist_plot
from ccbdl.learning.classifier import BaseClassifierLearning
from ccbdl.utils import DEVICE
import sys
import time


class Learner(BaseClassifierLearning):
    def __init__(self,
                 result_folder,
                 model,
                 train_data,
                 test_data,
                 val_data,
                 config,
                 network_config,
                 logging):
        """
        init function of the learner class.
        
        Args:
            model : The network that you use.
                --> Example CNN, FNN, etc.
            
            train_data, test_data val_data: Respective 
            train, test data and val_data.
                --> Example like Cifar10's train
                    and test data.

        Returns
            None.
        """
        super(Learner, self).__init__(train_data, test_data,
                                      val_data, result_folder, config, logging=logging)
        self.device = DEVICE
        print(self.device)
        
        self.model = model
        self.learning_rate = 10**self.learning_rate_exp
        self.learning_rate_l = 10**self.learning_rate_exp_l
        
        self.figure_storage.dpi=200
        
        if self.weight_decay_rate == 0:
            self.weight_decay = 0
        else:
            self.weight_decay = 10**self.weight_decay_rate
        
        self.learner_config = config
        self.network_config = network_config

        self.criterion = getattr(torch.nn, self.criterion)().to(self.device)
        
        # Get the last layer's name
        last_layer_name_parts = list(self.model.named_parameters())[-1][0].split('.')
        last_layer_name = last_layer_name_parts[0] + '.' + last_layer_name_parts[1]
        # print("Last layer name:", last_layer_name)
        
        # Separate out the parameters based on the last layer's name
        fc_params = [p for n, p in self.model.named_parameters() if last_layer_name + '.' in n]  # Parameters of the last layer
        rest_params = [p for n, p in self.model.named_parameters() if not last_layer_name + '.' in n]  # Parameters of layers before the last layer

        self.optimizer = getattr(torch.optim, self.optimizer)(
            rest_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        
        self.optimizer_fc = torch.optim.Adam(fc_params, lr=self.learning_rate_l)
        
        # print("FC Params:")
        # for p in fc_params:
        #     print(p.shape)
        # print("\nRest Params:")
        # for p in rest_params:
        #     print(p.shape)
        
        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_name)(
            self.optimizer, 
            milestones=[self.scheduler_step_1,self.scheduler_step_2,self.scheduler_step_3], 
            gamma=self.learning_rate_decay)
        
        self.scheduler_fc = getattr(torch.optim.lr_scheduler, self.scheduler_name)(
            self.optimizer_fc, 
            milestones=[self.scheduler_step_1,self.scheduler_step_2,self.scheduler_step_3], 
            gamma=self.learning_rate_decay)

        self.result_folder = result_folder

        # if sys.platform == "linux":
        self.plotter.register_default_plot(TimePlot(self))
        self.plotter.register_default_plot(Accuracy_plot(self))
        self.plotter.register_default_plot(Precision_plot(self))
        self.plotter.register_default_plot(Recall_plot(self))
        self.plotter.register_default_plot(F1_plot(self))
        
        if sys.platform == "win32":
            self.plotter.register_default_plot(Attribution_plots(self)) 
            
        self.plotter.register_default_plot(Loss_plot(self))
        self.plotter.register_default_plot(Learning_rate_plot(self))
        
        if sys.platform == "win32":
            self.plotter.register_default_plot(Softmax_plot(self))
            
        #if self.network_config["final_layer"] == 'nlrl':
            #self.plotter.register_default_plot(Hist_plot(self))
            
        if sys.platform == "win32":
            self.plotter.register_default_plot(Tsne_plot(self))

        self.parameter_storage.store(self)
        self.parameter_storage.write_tab(self.model.count_parameters(), "number of parameters: ")
        self.parameter_storage.write_tab(self.model.count_learnable_parameters(), 
                                         "number of learnable parameters: ")
        
        self.initial_save_path = os.path.join(self.result_folder, 'net_initial.pt')
        
        # for name, param in model.named_parameters():
        #     print(name)
        
        # Replace DataStorage store method with store_new for calculating correct a_train_Acc and a_train_loss
        self.data_storage.store = self.store_new

    def _train_epoch(self, train=True):
        if self.logging:
            self.logger.info("started epoch %i." % self.epoch)
        
        if self.epoch == 0:
            torch.save({'epoch': self.epoch,
                        'batch': 0,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'optimizer_fc_dict': self.optimizer_fc.state_dict()},
                       self.initial_save_path)

        self.model.train()

        for i, data in enumerate(self.train_data):            
            inputs, labels = data
            inputs, labels = inputs.to(
                self.device), labels.to(self.device).long()

            self.optimizer.zero_grad()
            self.optimizer_fc.zero_grad()

            outputs = self._classify(inputs)

            self.train_loss = self.criterion(outputs, labels)

            if train:
                self.train_loss.backward()
                self.optimizer.step()
                self.optimizer_fc.step()

            _, predicted = torch.max(outputs.data, 1)
            self.train_accuracy = sum((predicted == labels)/len(inputs))*100
            
            _, precision, recall, f1 = ccbdl.evaluation.plotting.classify.get_classification_scores(
            predicted, labels)
            
            self.train_precision = precision
            self.train_recall = recall
            self.train_f1 = f1

            self.data_storage.store([self.epoch, self.batch, self.train_loss,
                                    self.train_accuracy, self.test_loss, self.test_accuracy])
            self.data_storage.dump_store(
                "train_predictions", predicted.detach().cpu())
            self.data_storage.dump_store("train_labels", labels.detach().cpu())
            self.data_storage.dump_store("train_prec", self.train_precision)
            self.data_storage.dump_store("train_rec", self.train_recall)
            self.data_storage.dump_store("train_f1s", self.train_f1)
            self.data_storage.dump_store("test_prec", self.test_precision)
            self.data_storage.dump_store("test_rec", self.test_recall)
            self.data_storage.dump_store("test_f1s", self.test_f1)

            if train:
                self.batch += 1  
                
                # self.data_storage.dump_store("train_inputs", inputs)
                # self.data_storage.dump_store("train_actual_label", labels)

                self.data_storage.dump_store(
                    "learning_rate", self.optimizer.param_groups[0]['lr'])
                self.data_storage.dump_store(
                    "learning_rate_l", self.optimizer_fc.param_groups[0]['lr'])

    def _test_epoch(self):        
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_data):
                images, labels = data
                images, labels = images.to(
                    self.device), labels.to(self.device).long()
                
                outputs = self._classify(images)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                _, precision, recall, f1 = ccbdl.evaluation.plotting.classify.get_classification_scores(
                predicted, labels)

                
                self.data_storage.dump_store(
                    "test_predictions", predicted.detach().cpu())
                self.data_storage.dump_store(
                    "test_labels", labels.detach().cpu())
                
                # self.data_storage.dump_store("test_inputs", images)
                # self.data_storage.dump_store("test_actual_label", labels)

        self.test_accuracy = 100 * correct / total
        self.test_loss = running_loss / (i + 1)
        self.test_precision = precision
        self.test_recall = recall
        self.test_f1 = f1

    def _validate_epoch(self):
        pass

    def _classify(self, ins):
        return self.model(ins)

    def _update_best(self):
        if self.test_accuracy > self.best_values["TestAcc"]:
            self.best_values = {"Epoch":           self.epoch,
                                "TestLoss":        self.test_loss,
                                "TestAcc":         self.test_accuracy,
                                "TrainLoss":       self.train_loss.item(),
                                "TrainAcc":        self.train_accuracy.item(),
                                "Batch":           self.batch}

            self.best_state_dict = self.model.state_dict()
            self.best_optimizer_dict = self.optimizer.state_dict()
            self.best_optimizer_fc_dict = self.optimizer_fc.state_dict()

    def evaluate(self):
        if self.logging:
            self.logger.info("evaluation")

        self.end_values = {"Epoch":           self.epoch,
                           "TestLoss":        self.test_loss,
                           "TestAcc":         self.test_accuracy,
                           "TrainLoss":       self.train_loss.item(),
                           "TrainAcc":        self.train_accuracy.item(),
                           "Batch":           self.batch}

    def _hook_every_epoch(self):
        if self.epoch == 0:
            self.init_values = {"Epoch":           self.epoch,
                                "TestLoss":        self.test_loss,
                                "TestAcc":         self.test_accuracy,
                                "TrainLoss":       self.train_loss.item(),
                                "TrainAcc":        self.train_accuracy.item(),
                                "Batch":           self.batch}
            
            torch.save({'epoch': self.epoch,
                        'batch': self.init_values["Batch"],
                        'test_acc': self.init_values["TestAcc"],
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'optimizer_fc_dict': self.optimizer_fc.state_dict()},
                       self.init_save_path)


        self.scheduler.step()
        self.scheduler_fc.step()
        
        self.data_storage.dump_store("epochs_gen", self.epoch)

        if self.epoch != 0:
            self.data_storage.dump_store(
                "learning_rate", self.optimizer.param_groups[0]['lr'])
            self.data_storage.dump_store(
                "learning_rate_l", self.optimizer_fc.param_groups[0]['lr'])
        
        if sys.platform == "linux":
            if self.epoch == self.learner_config["num_epochs"] - 1:
                torch.save({'epoch': self.best_values["Epoch"],
                            'batch': self.best_values["Batch"],
                            'test_acc': self.best_values["TestAcc"],
                            'model_state_dict': self.best_state_dict,
                            'optimizer_state_dict': self.best_optimizer_dict,
                            'optimizer_fc_dict': self.best_optimizer_fc_dict},
                           self.best_save_path)
                
                torch.save({'epoch': self.epoch,
                            'batch': self.batch,
                            'test_acc': self.test_accuracy,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'optimizer_fc_dict': self.optimizer_fc.state_dict()},
                           self.net_save_path)
            
    def _save(self):
        if self.logging:
            self.logger.info(
                "saving the models and values of initial, best and final")

        if sys.platform == "win32":
            torch.save({'epoch': self.best_values["Epoch"],
                        'batch': self.best_values["Batch"],
                        'test_acc': self.best_values["TestAcc"],
                        'model_state_dict': self.best_state_dict,
                        'optimizer_state_dict': self.best_optimizer_dict,
                        'optimizer_fc_dict': self.best_optimizer_fc_dict},
                       self.best_save_path)
            
            torch.save({'epoch': self.epoch,
                        'batch': self.batch,
                        'test_acc': self.test_accuracy,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'optimizer_fc_dict': self.optimizer_fc.state_dict()},
                       self.net_save_path)

        self.parameter_storage.store(self.init_values, "initial_values")
        self.parameter_storage.store(self.best_values, "best_values")
        self.parameter_storage.store(self.end_values, "end_values")
        self.parameter_storage.write("\n")
        
        torch.save(self.data_storage, os.path.join(self.result_folder, "data_storage.pt"))
    
    def _load_initial(self):
        checkpoint = torch.load(self.initial_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model
    
    def _load_best(self):
        checkpoint = torch.load(self.best_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        return self.model
    
    def store_new(self, vals, force=False):
        """
        New store method to replace the default store method for DataStorage.

        Parameters
        ----------
        vals : list of values
            List of values to be stored in the internal 'stored_values'-dictionary.\n
            Order has to be the same as given during initialization. Best used with \n
            int, float or torch.Tensor.
        force : int
            If given an integer it appends the values with the given batch number.

        Returns
        -------
        None.

        """
        data_storage = self.data_storage  # Reference to data_storage
        # save time when first storing
        if data_storage.batch == 0:
            data_storage.dump_values["TimeStart"] = time.time()
        if data_storage.batch % data_storage.step == 0 or force > 0:
            if len(data_storage.stored_values["Time"]) == 0:
                data_storage.stored_values["Time"] = [
                    (time.time() - data_storage.dump_values["TimeStart"]) / 60]
            else:
                data_storage.stored_values["Time"].append(
                    (time.time() - data_storage.dump_values["TimeStart"]) / 60.0)
            for col in range(1, data_storage.columns):
                name = data_storage.names[col]
                if name == "a_train_loss":
                    if len(data_storage.stored_values["train_loss"]) < data_storage.average_window:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_loss"]))
                    else:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_loss"][-data_storage.average_window:]))
                    data_storage.stored_values[name].append(avg)
                elif name == "a_train_acc":
                    if len(data_storage.stored_values["train_acc"]) < data_storage.average_window:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_acc"]))
                    else:
                        avg = torch.mean(torch.Tensor(data_storage.stored_values["train_acc"][-data_storage.average_window:]))
                    data_storage.stored_values[name].append(avg)
                else:
                    if type(vals[col - 1]) == torch.Tensor:
                        data_storage.stored_values[name].append(
                            vals[col - 1].cpu().detach().item())
                    else:
                        data_storage.stored_values[name].append(vals[col - 1])
    
            if data_storage.batch == 0:
                data_storage._get_head()
                data_storage._display()
                print("")
            else:
                if data_storage.batch % data_storage.show == 0 or force > 0:
                    data_storage._display()
                if data_storage.batch % data_storage.line == 0:
                    print("")
                if data_storage.batch % data_storage.header == 0:
                    data_storage._get_head()
        data_storage.batch += 1
