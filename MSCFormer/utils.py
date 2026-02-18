import cv2
import os
import numpy as np
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:08:19 2023

@author: Administrator
"""

from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score  
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

import numpy as np
import pandas as pd
import scipy
import os
import shutil


def load_data_evaluate(
    dir_path,
    n_sub,
    mode_evaluate="LOSO",
    chosen_nodes=(9, 10, 14, 15, 16, 20, 21, 22),
    # NEW: subject range used for LOSO training pool
    loso_subject_range=(1, 9),   # inclusive (first, last)
    # NEW: explicit left-out subject (can be inside or outside the range)
    loso_left_out=None,
    # NEW: subject-dependent split settings
    subject_test_ratio=0.2,
    seed=42,
):
    """
    Primary entry point.

    - mode_evaluate == "LOSO":
        Train = all subjects in loso_subject_range except the left-out subject (if it is in range).
        Test  = left-out subject (always).
        If loso_left_out is outside loso_subject_range, it will still be used as the ONLY test subject,
        and no subject is removed from the training range.

    - otherwise (subject-dependent):
        Loads ONE subject, shuffles all trials, then splits into train/test by subject_test_ratio.
    """
    if mode_evaluate == "LOSO":
        if loso_left_out is None:
            loso_left_out = n_sub  # keep existing behavior if you pass n_sub as the fold subject
        return load_data_LOSO(
            dir_path=dir_path,
            subject_left_out=loso_left_out,
            chosen_nodes=chosen_nodes,
            loso_subject_range=loso_subject_range,
        )
    else:
        return load_data_subject_dependent(
            dir_path=dir_path,
            n_sub=n_sub,
            chosen_nodes=chosen_nodes,
            test_ratio=subject_test_ratio,
            seed=seed,
        )


def load_data_subject_dependent(dir_path, n_sub, chosen_nodes, test_ratio=0.2, seed=42):
    """
    Subject-specific:
      - loads all trials for ONE person
      - randomizes trial order
      - train/test split
    """
    X, y = load_data(dir_path, chosen_nodes, n_sub)

    # shuffle before split (your requirement)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]

    n_total = X.shape[0]
    n_test = max(1, int(round(test_ratio * n_total)))
    n_train = n_total - n_test

    train_data = X[:n_train]
    train_label = y[:n_train]
    test_data = X[n_train:]
    test_label = y[n_train:]

    return train_data, train_label, test_data, test_label


def load_data_LOSO(dir_path, subject_left_out, chosen_nodes, loso_subject_range=(1, 9)):
    """
    LOSO (subject-independent):
      - Training pool subjects = [first..last] inclusive, except subject_left_out if it is within that range
      - Test subject = subject_left_out only (even if outside the range)
    """
    first_sub, last_sub = loso_subject_range
    if first_sub > last_sub:
        raise ValueError(f"Invalid loso_subject_range={loso_subject_range}. Must be (first <= last).")

    # Always load test subject (can be outside training range)
    X_test, y_test = load_data(dir_path, chosen_nodes, subject_left_out)

    X_train_parts = []
    y_train_parts = []

    for n_sub in range(first_sub, last_sub + 1):
        # If left-out subject is in the training range, exclude it from training
        if n_sub == subject_left_out:
            continue

        X_sub, y_sub = load_data(dir_path, chosen_nodes, n_sub)
        X_train_parts.append(X_sub)
        y_train_parts.append(y_sub)

    if len(X_train_parts) == 0:
        raise ValueError(
            "LOSO produced empty training set. "
            f"Check loso_subject_range={loso_subject_range} and subject_left_out={subject_left_out}."
        )

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)

    return X_train, y_train, X_test, y_test


def load_data(dir_path, chosen_nodes, n_sub):
    """
    Load all trials for a single subject from .pt files.

    Expected structure:
      dir_path/
        sub_{n_sub}/
          trial_00000.pt
          trial_00001.pt
          ...

    Each trial_*.pt is a dict with:
      "data": FloatTensor shape (60, 1000)
      "label": LongTensor scalar
    """
    subject_path = os.path.join(dir_path, f"sub_{n_sub}")

    if not os.path.isdir(subject_path):
        raise FileNotFoundError(f"Subject folder not found: {subject_path}")

    trial_files = sorted(f for f in os.listdir(subject_path) if f.endswith(".pt"))
    if len(trial_files) == 0:
        raise FileNotFoundError(f"No .pt trial files found in: {subject_path}")

    X_list, y_list = [], []

    for fname in trial_files:
        file_path = os.path.join(subject_path, fname)
        trial_dict = torch.load(file_path, map_location="cpu")

        data = trial_dict["data"]    # (60, 1000)
        label = trial_dict["label"]  # scalar

        data = data[chosen_nodes, :]  # (len(chosen_nodes), 1000)

        X_list.append(data.numpy())
        y_list.append(int(label.item()))

    X = np.stack(X_list, axis=0)  # (n_trials, n_channels, 1000)
    y = np.array(y_list)          # (n_trials,)

    return X, y




def calMetrics(y_true, y_pred):
    '''
    calcuate the metrics: accuracy, precison, recall, f1, kappa

    Parameters
    ----------
    y_true : numpy or Series or list
        ground true lable.
    y_pred : numpy or Series or list
        predict label.

    Returns
    -------
    accuracy : float
        accuracy.
    precison : float
        precison.
    recall : float
        recall.
    f1 : float
        F1 score value.
    kappa : float
        kappa value.

    '''
    number = max(y_true)
    if number == 2:
        mode = 'binary'
    else:
        mode = 'macro'
    
    accuracy = accuracy_score(y_true, y_pred)
    precison = precision_score(y_true, y_pred, average=mode)
    recall = recall_score(y_true, y_pred, average=mode)
    f1 = f1_score(y_true, y_pred, average=mode)
    kappa = cohen_kappa_score(y_true, y_pred)
    return accuracy, precison, recall, f1, kappa
    


def calculatePerClass(data_dict, metric_name='Precision'):
    '''
    Calculate the performance metrics for each category

    Parameters
    ----------
    data_dict : dict
        Contains data for all subjects：{'1': DataFrame, '2':DataFrame ...}.
    metric_name : str, optional
        The value is in ['Precision', 'Recalll']. The default is 'Precision'.

    Returns
    -------
    df: DataFrame
        Calculation results of the specified metrics for all categories across all subjects

    '''
    metric_dict = {}
    for key in data_dict.keys():
        df = data_dict[key]
        if metric_name == 'Precision':
            metric_dict[key] = precision_score(df['true'], df['pred'], average=None)
        elif metric_name == 'Recall':
            metric_dict[key] = recall_score(df['true'], df['pred'], average=None)
    df = pd.DataFrame(metric_dict)
    df = df*100
    df = df.applymap(lambda x: round(x, 2))
    mean = df.apply('mean', axis=1).round(2) 
    std  = df.apply('std', axis=1).round(2) 
    df['mean'] = mean
    df['std'] = std
    df['metrics'] = metric_name
    return df



def numberClassChannel(database_type):
    if database_type=='A':
        number_class = 4
        number_channel = 22
    elif database_type=='B':
        number_class = 2
        number_channel = 3
    elif database_type=='C':
        number_class = 12
        number_channel = 8
    return number_class, number_channel




#
#The following code is derived from this open-source code：https://github.com/eeyhsong/EEG-Conformer/blob/main/visualization/utils.py
#

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # Forward propagation yields the network output logits (before applying softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            output=output[1]
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        print('the loss is', loss)
        loss.backward(retain_graph=True)
        # loss.backward(torch.ones_like(output), retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap  # + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img