from dataset import coast
from model import MSNet
import matplotlib.pyplot as plt
from torch.optim import Adam
import select_bands
import torch
import utils
import metric
import os
from SeT import (
    TotalLoss,
    Mask,
    separation_training
)

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lmda = 1e-3
num_bs = 64
num_layers = 3
lr = 1e-3
epochs = 150
output_iter = 5
max_iter = 10
data_norm = True
Net = MSNet
net_kwargs = dict()
net_kwargs['num_layers'] = num_layers

# Load data
dataset = coast
data, gt = dataset.get_data()
rows, cols, bands = data.shape
net_kwargs['shape'] = (rows, cols, num_bs)
print('Detecting on %s...' % dataset.name)

# Preprocessing
band_idx = select_bands.OPBS(data, num_bs)
data_bs = data[:, :, band_idx]
if data_norm:
    data_bs = utils.ZScoreNorm().fit(data_bs).transform(data_bs)

# Load model
model = Net(**net_kwargs).to(device).float()

# Loss
loss = TotalLoss(lmda, device)

# Mask
mask = Mask((rows, cols), device)

# Optimizer
optimizer = Adam(model.parameters(), lr=lr)

# Separation Training
x_bs = torch.from_numpy(data_bs).to(device).float()
pr_dm, history = separation_training(
    x=x_bs,
    gt=gt,
    model=model,
    loss=loss,
    mask=mask,
    optimizer=optimizer,
    epochs=epochs,
    output_iter=output_iter,
    max_iter=max_iter,
    verbose=True
)

# Save the detection result
result_path = os.path.join('results', model.name)
if not os.path.exists(result_path):
    os.makedirs(result_path)

rx_dm = utils.rx(data)
fpr, tpr, rx_auc = metric.roc_auc(rx_dm, gt)
plt.plot(fpr, tpr, label='RX: %.4f' % rx_auc)

fpr, tpr, pr_auc = metric.roc_auc(pr_dm, gt)
plt.plot(fpr, tpr, label='%s+SeT: %.4f' % (model.name, pr_auc),
         c='black', alpha=0.7)

plt.grid(alpha=0.3)
plt.legend()
plt.savefig(
    os.path.join(result_path, '%s_roc.pdf' % dataset.name)
)
plt.clf()
plt.close()

iters = [(_ + 1) * epochs for _ in range(max_iter)]
plt.xticks(iters)
plt.plot(iters, history)
plt.scatter([output_iter * epochs], [history[output_iter - 1]],
            marker='o', edgecolors='black', facecolors='white', label='Stop',
            zorder=10)
plt.grid(alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.savefig(
    os.path.join(result_path, '%s_auc_history.pdf' % dataset.name)
)
plt.clf()
plt.close()

print('Complete.')
print('Results are saved in results/.')


