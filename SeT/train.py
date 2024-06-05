import torch
from .detect import detect
from tqdm import tqdm
from metric import roc_auc
import numpy as np


def train_model(
        x,
        model,
        criterion,
        cri_kwargs,
        epochs,
        optimizer,
        verbose):

    epoch_iter = iter(_ for _ in range(epochs))
    if verbose:
        epoch_iter = tqdm(list(epoch_iter))
    for _ in epoch_iter:

        # Clear gradient information
        optimizer.zero_grad()

        # Forward propagation
        y = model(x)

        # Calculate loss
        loss = criterion(x=x, y=y, **cri_kwargs)

        # Backward propagation
        loss.backward()

        # Update network parameters
        optimizer.step()

        if verbose:
            epoch_iter.set_postfix({'loss': '{0:.4f}'.format(loss)})


def separation_training(
        x: torch.Tensor,
        gt: np.ndarray,
        model,
        loss,
        mask,
        optimizer,
        epochs,
        output_iter,
        max_iter,
        verbose) -> (np.ndarray, list):
    """
    The main process of the separation training algorithm.

    """

    history = []
    output_dm = np.zeros_like(gt)

    for i in range(1, max_iter + 1):
        if verbose:
            print('Iter {0}'.format(i))

        # Feed the model with x
        model_input = x

        # Train the model for some epochs
        train_model(
            model_input,
            model,
            loss,
            {'mask': mask},
            epochs,
            optimizer,
            verbose
        )

        # Update the mask using detection map obtained in this iteration
        dm = detect(x, model(model_input))
        mask.update(dm.detach())

        # Evaluation
        np_dm = dm.cpu().detach().numpy()
        fpr, tpr, auc = roc_auc(np_dm, gt)
        if verbose:
            print('Current AUC score: {0:.4f}'.format(auc))

        # Record history
        history.append(auc)

        # Record the output detection map of the algorithm
        if i == output_iter:
            output_dm = np_dm

    return output_dm, history
