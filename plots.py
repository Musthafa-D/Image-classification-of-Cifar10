import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from captum.attr import Saliency, GuidedBackprop, InputXGradient, Deconvolution, Occlusion
from ccbdl.utils.logging import get_logger
from ccbdl.evaluation.plotting.base import GenericPlot
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
from networks import NLRL_AO
from sklearn.manifold import TSNE
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
import random
from setsize import set_size
import matplotlib.ticker as ticker


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


class Accuracy_plot(GenericPlot):
    def __init__(self, learner):
        super(Accuracy_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating accuracy plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_acc")
        yatr = self.learner.data_storage.get_item("a_train_acc")
        yt = self.learner.data_storage.get_item("test_acc")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathrm{Acc}_{\\mathrm{train}}$')
        ax.plot(x, yatr, label='$\\mathrm{Acc}_{\\mathrm{train\\_avg}}$')
        ax.plot(x, yt, label='$\\mathrm{Acc}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Acc}$", fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=100)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.set_yticks(range(10, 101, 10))
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "accuracies"))
        return figs, names


class Precision_plot(GenericPlot):
    def __init__(self, learner):
        super(Precision_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating precision plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_prec")
        yt = self.learner.data_storage.get_item("test_prec")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        
        ax.plot(x, ytr, label='$\\mathrm{Prec}_{\\mathrm{train}}$')
        ax.plot(x, yt, label='$\\mathrm{Prec}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Prec}$", fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=1.05)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "precisions"))
        return figs, names


class Recall_plot(GenericPlot):
    def __init__(self, learner):
        super(Recall_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating recall plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_rec")
        yt = self.learner.data_storage.get_item("test_rec")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathrm{Rec}_{\\mathrm{train}}$')
        ax.plot(x, yt, label='$\\mathrm{Rec}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{Rec}$", fontsize=14)
        
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=1.05)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "recalls"))
        return figs, names


class F1_plot(GenericPlot):
    def __init__(self, learner):
        super(F1_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating f1 plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_f1s")
        yt = self.learner.data_storage.get_item("test_f1s")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathrm{F1}_{\\mathrm{train}}$')
        ax.plot(x, yt, label='$\\mathrm{F1}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel("$\\mathrm{F1}$", fontsize=14)
        
        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=1.05)

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "f1s"))
        return figs, names


class Loss_plot(GenericPlot):
    def __init__(self, learner):
        super(Loss_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating loss plot")

    def consistency_check(self):
        return True

    def plot(self):
        x = self.learner.data_storage.get_item("batch")
        ytr = self.learner.data_storage.get_item("train_loss")
        yatr = self.learner.data_storage.get_item("a_train_loss")
        yt = self.learner.data_storage.get_item("test_loss")

        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))

        ax.plot(x, ytr, label='$\\mathcal{L}_{\\mathrm{train}}$')
        ax.plot(x, yatr, label='$\\mathcal{L}_{\\mathrm{train\\_avg}}$')
        ax.plot(x, yt, label='$\\mathcal{L}_{\\mathrm{test}}$')

        ax.set_xlabel("$B$", fontsize=14)
        ax.set_ylabel('$\\mathcal{L}$', fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0, top=max(ytr))

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.grid(True)
        ax.set_yticks(range(1, 3, 1))
        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "losses"))
        return figs, names


class TimePlot(GenericPlot):
    def __init__(self, learner):
        super(TimePlot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating time plot")

    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        
        xs, ys = zip(*self.learner.data_storage.get_item("Time", batch=True))
        
        ax.plot(xs, [y - ys[0]for y in ys])
        ax.set_xlabel('$B$', fontsize=14)
        ax.set_ylabel('$t$', fontsize=14)
        
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=max(ys))

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "time_plot"))    
        return figs, names


class Learning_rate_plot(GenericPlot):
    def __init__(self, learner):
        super(Learning_rate_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating learning rate plot")

    def consistency_check(self):
        return True

    def plot(self):
        figs = []
        names = []
        
        learning_rate = self.learner.data_storage.get_item("learning_rate")
        learning_rate_l = self.learner.data_storage.get_item("learning_rate_l")
        x = self.learner.data_storage.get_item("batch")

        fig, ax = plt.subplots(1, 1, figsize=set_size('thesis'))
        ax.plot(learning_rate, label="$\\mathrm{lr}_{\\mathrm{m}}$")
        ax.plot(learning_rate_l, label="$\\mathrm{lr}_{\\mathrm{f}}$")

        ax.set_xlabel('$B$', fontsize=14)
        ax.set_ylabel('$lr$', fontsize=14)

        ax.set_xlim(left=0, right=max(x))
        ax.set_ylim(bottom=0)

        # Use ScalarFormatter to display y-axis in scientific notation (powers of 10)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.legend()

        figs.append(fig)
        plt.close(fig)
        names.append(os.path.join("plots", "learning_rate_schedule"))
        return figs, names


class Softmax_plot(GenericPlot):
    def __init__(self, learner):
        super(Softmax_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating softmax bar plots")

    def consistency_check(self):
        return True
    
    def values(self, types):
        inputs_list = self.learner.data_storage.get_item(f"{types}_inputs")
        labels_list = self.learner.data_storage.get_item(f"{types}_actual_label")
        
        inputs = torch.cat(inputs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        # Dictionary to hold indices for each label
        label_indices = {i: [] for i in range(10)}  # labels from 0 to 9
        
        # Populate the dictionary with indices for each label
        for idx, label in enumerate(labels):
            label_indices[label.item()].append(idx)
        
        # List to store the selected indices
        selected_indices = []
    
        # Sample 5 images from each label group
        for label, indices in label_indices.items():
            if len(indices) >= 5:
                selected_indices.extend(random.sample(indices, 5))
            else:
                selected_indices.extend(indices)
        
        # Extract the selected inputs and labels
        selected_inputs = inputs[selected_indices]
        selected_labels = labels[selected_indices]
        return selected_inputs, selected_labels

    def plot(self):
        names = []
        figs = []
        
        max_images_per_plot = 5  # Define a constant for the maximum number of images per plot

        for types in ["train", "test"]:
            inputs, labels = self.values(types)
            sm = torch.nn.Softmax(dim=0)
            
            class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                          6: "frog", 7: "horse", 8: "ship", 9: "truck"}
            
            for models in ["initial", "best"]:   
                if models == "initial":
                    model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
                else:
                    model = self.learner._load_best()  # Load the best epoch's model with the respective weights
                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                # Group indices by class and correctness
                correct_indices = {i: [] for i in range(10)}
                incorrect_indices = {i: [] for i in range(10)}

                for i in range(inputs.shape[0]):
                    if labels[i] == preds[i]:
                        correct_indices[labels[i].item()].append(i)
                    else:
                        incorrect_indices[labels[i].item()].append(i)

                for correct, indices_dict in zip([True, False], [correct_indices, incorrect_indices]):
                    for class_idx, indices in indices_dict.items():
                        if indices:
                            subsets = [indices[x:x + max_images_per_plot] for x in range(0, len(indices), max_images_per_plot)]
    
                            for subset in subsets:
                                num_rows = 2
                                num_cols = len(subset)
                                fig, axs = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 8), squeeze=False)
    
                                for idx, image_idx in enumerate(subset):
                                    img = (inputs[image_idx].cpu().detach().permute(1, 2, 0)).numpy()
                                    label = labels[image_idx].cpu().detach()
                                    pred = preds[image_idx].cpu().detach()
    
                                    num_classes = outputs[image_idx].cpu().detach().shape[0]
                                    output_softmax = sm(outputs[image_idx].cpu().detach()).numpy()
    
                                    axs[0, idx].imshow(img)
                                    axs[0, idx].set_title(f"Actual: {class_dict[label.item()]}\nPredicted: {class_dict[pred.item()]}", fontsize=17)
                                    axs[0, idx].axis("off")
                                    
                                    axs[1, idx].bar(range(num_classes), output_softmax)
                                    axs[1, idx].set_xticks(range(num_classes))
                                    if idx == 0:
                                        axs[1, idx].set_ylabel("$P$", fontsize=17) # class probability P(y/x)
                                    if len(subset) == 5:
                                        axs[1, 2].set_xlabel("$y$", fontsize=17) # class
                                    if len(subset) == 4:
                                        axs[1, 1].set_xlabel("$y$", fontsize=17)
                                    if len(subset) == 3:
                                        axs[1, 1].set_xlabel("$y$", fontsize=17)
                                    if len(subset) == 2:
                                        axs[1, 0].set_xlabel("$y$", fontsize=17)
                                    if len(subset) == 1:
                                        axs[1, idx].set_xlabel("$y$", fontsize=17)
                                    axs[1, idx].set_ylim((0, 1))
                                    axs[1, idx].set_yticks(torch.arange(0, 1.1, 0.1).tolist())
                                
                                if correct:
                                    names.append(os.path.join("plots", "analysis_plots", "softmax_plots", f"{models}_{types}_correctly_classified_class_{class_idx}_{subsets.index(subset) + 1}"))
                                else:
                                    names.append(os.path.join("plots", "analysis_plots", "softmax_plots", f"{models}_{types}_misclassified_class_{class_idx}_{subsets.index(subset) + 1}"))
                                
                                figs.append(fig)
                                plt.close(fig)
        return figs, names


class Attribution_plots(GenericPlot):
    def __init__(self, learner):
        super(Attribution_plots, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating attribution maps")

    def consistency_check(self):
        return True

    def safe_visualize(self, attr, title, fig, ax, label, img_name, types, cmap, check):
        if not (attr == 0).all():
            if len(attr.shape) == 2:
                attr = np.expand_dims(attr, axis=2)
            if check == 0:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         title=title,
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
            else:
                viz.visualize_image_attr(attr,
                                         method='heat_map',
                                         sign='all',
                                         plt_fig_axis=(fig, ax),
                                         use_pyplot=False,
                                         fontsize=17,
                                         cmap=cmap)
        else:
            print(f"Skipping visualization for {types} data's label: {label}, {img_name} for the attribution: {title} as all attribute values are zero.")

    def plot(self):
        names = []
        figs = []
        max_images_per_plot = 5  # Define a constant for the maximum number of images per plot
        
        for models in ["initial", "best"]:
            if models == "initial":
                model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                model = self.learner._load_best()  # Load the best epoch's model with the respective weights
                
            cmap = LinearSegmentedColormap.from_list("BlWhGn", ["blue", "white", "green"])
    
            imp_values = Softmax_plot(self.learner)
    
            for types in ["train", "test"]:
                inputs, labels = imp_values.values(types)
                inputs.requires_grad = True  # Requires gradients set true
    
                class_dict = {0: "plane", 1: "car", 2: "bird", 3: "cat", 4: "deer", 5: "dog",
                              6: "frog", 7: "horse", 8: "ship", 9: "truck"}
                
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps = attribution_maps(model, inputs, preds)
    
                attrs = ["Saliency", "Guided Backprop", "Input X Gradient", "Deconvolution", "Occlusion"]
    
                # Group indices by class and correctness
                correct_indices = {i: [] for i in range(10)}
                incorrect_indices = {i: [] for i in range(10)}

                for i in range(inputs.shape[0]):
                    if labels[i] == preds[i]:
                        correct_indices[labels[i].item()].append(i)
                    else:
                        incorrect_indices[labels[i].item()].append(i)

                for correct, indices_dict in zip([True, False], [correct_indices, incorrect_indices]):
                    for class_idx, indices in indices_dict.items():
                        if indices:
                            subsets = [indices[x:x + max_images_per_plot] for x in range(0, len(indices), max_images_per_plot)]
    
                            for subset in subsets:
                                num_rows = len(subset)
                                num_cols = len(attrs) + 1
                                fig, axs = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 4 * num_rows))
                                if num_rows == 1 and num_cols == 1:
                                    axs = np.array([[axs]])
                                elif num_rows == 1:
                                    axs = axs[np.newaxis, :]
                                elif num_cols == 1:
                                    axs = axs[:, np.newaxis]
    
                                count = 0
                                for idx in subset:
                                    img = (inputs[idx].cpu().detach().permute(1, 2, 0)).numpy()
    
                                    label = labels[idx].cpu().detach()
                                    pred = preds[idx].cpu().detach()
                                    
                                    results = [
                                        np.transpose(saliency_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        np.transpose(guided_backprop_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        np.transpose(input_x_gradient_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        np.transpose(deconv_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                        np.transpose(occlusion_maps[idx].squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                    ]
    
                                    axs[count, 0].imshow(img)
                                    axs[count, 0].set_title(f"Actual: {class_dict[label.item()]}\nPredicted: {class_dict[pred.item()]}", fontsize=17)
                                    axs[count, 0].axis("off")
    
                                    for col, (attr, res) in enumerate(zip(attrs, results)):
                                        title = f"{attr}"
                                        if len(subset) > 1:
                                            if idx == subset[0]:
                                                check = 0
                                            else:
                                                check = 1
                                        else:
                                            check = 0
                                        self.safe_visualize(res, title, fig, axs[count, col + 1], label, class_dict[label.item()], types, cmap, check)
    
                                    count += 1
                                
                                # Add a single colorbar for all subplots below the grid
                                fig.subplots_adjust(bottom=0.15)
                                cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
                                norm = plt.Normalize(vmin=-1, vmax=1)
                                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                                sm.set_array([])
                                cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                                cbar.ax.tick_params(labelsize=17)  # Set colorbar tick label font size
    
                                if correct:
                                    names.append(os.path.join("plots", "analysis_plots", "attribution_plots", f"{models}_{types}_correctly_classified_class_{class_idx}_{subsets.index(subset) + 1}"))
                                else:
                                    names.append(os.path.join("plots", "analysis_plots", "attribution_plots", f"{models}_{types}_misclassified_class_{class_idx}_{subsets.index(subset) + 1}"))
                
                                figs.append(fig)
                                plt.close(fig)
        return figs, names


class Hist_plot(GenericPlot):
    def __init__(self, learner):
        super(Hist_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating histogram of nlrl_ao plot")

    def consistency_check(self):
        return True
    
    def extract_parameters(self, model):
        for layer in model.modules():
            if isinstance(layer, NLRL_AO):
                negation = torch.sigmoid(layer.negation).detach().cpu().numpy()
                relevancy = torch.sigmoid(layer.relevancy).detach().cpu().numpy()
                selection = torch.sigmoid(layer.selection).detach().cpu().numpy()
    
                negation_init = torch.sigmoid(layer.negation_init).detach().cpu().numpy()
                relevancy_init = torch.sigmoid(layer.relevancy_init).detach().cpu().numpy()
                selection_init = torch.sigmoid(layer.selection_init).detach().cpu().numpy()
                
                return (negation, relevancy, selection), (negation_init, relevancy_init, selection_init)
        return None


    def plot(self):
        figs=[]
        names=[]
        
        bool_ops = ['negation', 'relevancy', 'selection']
        # Load the classifier  
        for models in ["initial", "best"]:
            if models == "initial":
                model = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                model = self.learner._load_best()  # Load the best epoch's model with the respective weights
            params, init_params = self.extract_parameters(model)
        
            for i, (param, init_param) in enumerate(zip(params, init_params)):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(init_param.ravel(), color='blue', alpha=0.5, bins=np.linspace(0, 1, 30), label='Initial')
                ax.hist(param.ravel(), color='red', alpha=0.5, bins=np.linspace(0, 1, 30), label='Trained')
                
                ax.set_xlabel('$\sigma(W)$', fontsize=14) # sigmoid of the learnable weight matrices
                ax.set_ylabel('$|W|$', fontsize=14) # number of parameters
                ax.legend(loc='upper right')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                
                figs.append(fig)
                plt.close(fig)
                names.append(os.path.join("plots", "histogram_plots", f"{models}_{bool_ops[i]}"))
        return figs, names


# Other functions
def attributions(model, inputs, labels):
    # Initialize the Saliency object
    saliency = Saliency(model)
    # Initialize the GuidedBackprop object
    guided_backprop = GuidedBackprop(model)
    # Initialize the DeepLift object
    input_x_gradient = InputXGradient(model)
    # Initialize the Deconvolution object
    deconv = Deconvolution(model)
    # Initialize the Occlusion object
    occlusion = Occlusion(model)  
    return saliency, guided_backprop, input_x_gradient, deconv, occlusion

def attribution_maps(model, inputs, labels):
    saliency, guided_backprop, input_x_gradient, deconv, occlusion = attributions(model, inputs, labels)
    
    saliency_maps = saliency.attribute(inputs, target=labels)
    guided_backprop_maps = guided_backprop.attribute(inputs, target=labels)
    input_x_gradient_maps = input_x_gradient.attribute(inputs, target=labels)
    deconv_maps = deconv.attribute(inputs, target=labels)
    occlusion_maps = occlusion.attribute(inputs, target=labels, sliding_window_shapes=(3, 5, 5), strides=(3, 2, 2))   
    return saliency_maps, guided_backprop_maps, input_x_gradient_maps, deconv_maps, occlusion_maps


class Tsne_plot(GenericPlot):
    def __init__(self, learner):
        super(Tsne_plot, self).__init__(learner, repeat=0)
        self.logger = get_logger()
        self.logger.info("creating tsne plots based on classifier's features and classifier's decision")
    
    def consistency_check(self):
        return True
    
    def get_features(self, classifier, imgs):
        activation = {}
        
        def get_activation(name):
            def hook(classifier, inp, output):
                activation[name] = output.detach()
            return hook
        
        # Register the hook
        if self.learner.network_config["final_layer"] == 'nlrl':
            handle = classifier.model[-2].register_forward_hook(get_activation('conv'))
        else:
            handle = classifier.model[-1].register_forward_hook(get_activation('conv'))
        _ = classifier(imgs)
        
        # Remove the hook
        handle.remove()
        return activation['conv']
    
    def compute_tsne(self, features):
        tsne = TSNE(n_components=2, random_state=0)
        tsne_results = tsne.fit_transform(features)
        return tsne_results
    
    def process_images(self, data_loader, classifier, cat):
        all_features = []
        all_labels = []
        
        for imgs in data_loader:
            outputs = classifier(imgs)
            _, predicted_labels = torch.max(outputs, 1)
            features = self.get_features(classifier, imgs)
            features = features.view(features.size(0), -1)  # Flatten the features
            all_features.append(features)
            all_labels.append(predicted_labels)
            
        # Concatenate all the features and labels from the batches
        if cat == 1:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        return all_features, all_labels
    
    def plot(self):
        figs, names = [], []
        # Load the classifier  
        for models in ["initial", "best"]:
            if models == "initial":
                classifier = self.learner._load_initial()  # Load the initial epoch's model with the respective weights
            else:
                classifier = self.learner._load_best()  # Load the best epoch's model with the respective weights
        
            # Setting concatenation true by initializing value as 1
            cat = 1
            
            epochs = self.learner.data_storage.get_item("epochs_gen")
            total = len(epochs)
        
            for types in ["train", "test"]:
                total_images = self.learner.data_storage.get_item(f"{types}_inputs")
                batches_per_epoch = int(len(total_images)/total)
                
                images = total_images[-batches_per_epoch:]
                images = torch.cat(images)
                dataset = ImageTensorDataset(images)
                data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
                features, labels = self.process_images(data_loader, classifier, cat)
                
                label_counts = [torch.sum(labels == i).item() for i in range(10)]
            
                tsne_results = self.compute_tsne(features.cpu().numpy())
                
                # Define a color palette for the labels
                palette = sns.color_palette("colorblind", 10)
            
                # Plotting
                for label in range(10):  # Cifar10 dataset has 10 labels
                    fig, ax = plt.subplots(figsize=(16, 10))
                    # images scatter plot
                    indices = (labels == label).cpu().numpy()
                    sns.scatterplot(
                        ax=ax, 
                        x=tsne_results[indices, 0], 
                        y=tsne_results[indices, 1], 
                        label=f"{label}", 
                        color=palette[label],
                        alpha=0.5
                    )
                    ax.legend()
                    figs.append(fig)
                    plt.close(fig)
                    names.append(os.path.join("plots", "analysis_plots", "tsne_plots", f"{models}_{types}_label_{label}_counts_{label_counts[label]}"))
            
                fig, ax = plt.subplots(figsize=(16, 10))
                
                for label in range(10):  # Cifar10 dataset has 10 labels
                    # Filter data points by label
                    indices = (labels == label).cpu().numpy()
                    sns.scatterplot(
                        ax=ax, 
                        x=tsne_results[indices, 0], 
                        y=tsne_results[indices, 1], 
                        label=f"{label}", 
                        color=palette[label],
                        alpha=0.5
                    )
                ax.legend()
                
                figs.append(fig)  
                plt.close(fig)
                names.append(os.path.join("plots", "analysis_plots", "tsne_plots", f"{models}_{types}_combined"))               
        return figs, names


# Custom Dataset class to handle lists of tensors
class ImageTensorDataset(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]
