import os
import torch
from tqdm.notebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import time

from lib.utils.plotters import torch_to_img, _plot_img, _plot_score, _plot_hist

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def compute_scores(gt, pred, eps=1e-6):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    pred = torch.clamp(pred, 0) + eps
    gt = gt + eps
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())
    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def score(model, data, criterion, is_train_data, desc):
    model.eval()
    metrics = [0, 0, 0, 0, 0, 0, 0]  # abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    loss = 0

    if is_train_data:
        N = max(1, int(len(data) * 0.1))
    else:
        N = len(data)

    data_iter = iter(data)

    for _ in tqdm(range(N), desc=desc):
        rgb, depth, distr = next(data_iter)
        rgb, depth, distr = rgb.to(device), depth.to(device), distr.to(device)
        with torch.no_grad():
            pred = model(rgb)
        errors = compute_scores(depth, pred)
        metrics = [m + e for m, e in zip(metrics, errors)]
        if criterion is not None:
            loss += criterion(depth, pred, rgb) / N
    scores = [m / N for m in metrics]
    if criterion is not None:
        scores.append(loss)
    return scores


def evaluate_and_log_history(history, model, criterion, data_tr, data_vl):
    model.eval()
    train_scores = score(model, data_tr, criterion, is_train_data=True, desc='Scoring train')
    val_scores = score(model, data_vl, criterion, is_train_data=False, desc='  Scoring val')

    history['abs_rel']['train'].append(train_scores[0].item())
    history['sq_rel']['train'].append(train_scores[1].item())
    history['rmse']['train'].append(train_scores[2].item())
    history['rmse_log']['train'].append(train_scores[3].item())
    history['a1']['train'].append(train_scores[4].item())
    history['a2']['train'].append(train_scores[5].item())
    history['a3']['train'].append(train_scores[6].item())
    history['losses']['train'].append(train_scores[7].item())

    history['abs_rel']['val'].append(val_scores[0].item())
    history['sq_rel']['val'].append(val_scores[1].item())
    history['rmse']['val'].append(val_scores[2].item())
    history['rmse_log']['val'].append(val_scores[3].item())
    history['a1']['val'].append(val_scores[4].item())
    history['a2']['val'].append(val_scores[5].item())
    history['a3']['val'].append(val_scores[6].item())
    history['losses']['val'].append(val_scores[7].item())

    return history


def draw_results(history, model, data):
    clear_output(wait=True)

    plt.figure(figsize=(18, 10))
    epoch = len(history['losses']['train'])
    plt.suptitle('Epoch: %d/%d; %s; %s = %f' % (
        epoch,
        history['info']['epochs'],
        history['info']['model'],
        history['info']['criterion'],
        history['losses']['train'][-1]
    )
                 )

    rgb, depth, _ = next(iter(data))

    model.eval()
    with torch.no_grad():
        pred = model(rgb.to(device))

    N = min(rgb.shape[0], 4)
    rgb = torch_to_img(rgb[:N, :, :, :])
    depth = depth[:N, 0, :, :].detach().cpu().numpy()
    pred = pred[:N, 0, :, :].detach().cpu().numpy()

    for k in range(N):
        _plot_img(k, rgb[k], 'Input', 0)
        _plot_img(k, pred[k], 'Out [%.2f; %.2f]' % (np.min(pred[k]), np.max(pred[k])), 1)
        _plot_img(k, depth[k], ' GT [%.2f; %.2f]' % (np.min(depth[k]), np.max(depth[k])), 2)


    _plot_hist(history['losses'], 'Loss', 1, 'log')

    _plot_score(history['abs_rel'], 'AbsRel ↓', 13)
    _plot_score(history['sq_rel'], 'SqRel ↓', 14)
    _plot_score(history['rmse'], 'RMSE ↓', 19)
    _plot_score(history['rmse_log'], 'RMSE_log ↓', 20)

    plt.show()


def save_checkpoint(model, optimizer, loss, epoch, info):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    filename = '%s.%s.pt' % (info['model'], info['criterion'])
    os.system('rm -rf %s' % filename)
    torch.save(state, filename)


class Trainer:
    def __init__(self, model, opt, criterion, criterion_distr, loss, epochs, data_tr, data_vl, distr_weight, scheduler):
        super().__init__()
        self.model = model
        self.opt = opt
        self.criterion = criterion
        self.criterion_d = criterion_distr
        self.epochs = epochs
        self.data_tr = data_tr
        self.data_vl = data_vl
        self.dw = distr_weight
        self.multiscale_loss = loss
        self.scheduler = scheduler

    def train(self):
        torch.cuda.empty_cache()

        history = {
            'losses': {'train': [], 'val': []},
            'abs_rel': {'train': [], 'val': []},
            'sq_rel': {'train': [], 'val': []},
            'rmse': {'train': [], 'val': []},
            'rmse_log': {'train': [], 'val': []},
            'a1': {'train': [], 'val': []},
            'a2': {'train': [], 'val': []},
            'a3': {'train': [], 'val': []},
            'time': [],
            'info': {
                'model': self.model.__class__.__name__,
                'criterion': self.criterion.__class__.__name__,
                'epochs': self.epochs
            }
        }

        t0 = time.time()
        best_val_loss = 10000000.0
        for epoch in range(self.epochs):

            start_time = time.time()
            train_loss = 0
            self.model.train()  # train mode

            for rgb, depth, distr in tqdm(self.data_tr, desc="Training Epoch"):
                rgb, depth, distr = rgb.to(device), depth.to(device), distr.to(device)
                self.opt.zero_grad()
                output = self.model(rgb)
                if torch.is_tensor(output):
                    loss = self.criterion(depth, output, rgb)
                else:
                    D0, D1, D2, D3, DD = output
                    loss = self.multiscale_loss(self.criterion, self.criterion_d, distr, depth, D0, D1, D2, D3, DD, rgb, self.dw)
                loss.backward()
                self.opt.step()
                train_loss += loss / len(self.data_tr)

            if self.scheduler is not None:
                self.scheduler.step()

            self.model.eval()

            history['time'].append(time.time() - start_time)
            history = evaluate_and_log_history(history, self.model, self.criterion, self.data_tr, self.data_vl)
            draw_results(history, self.model, self.data_vl)

            val_loss = history['losses']['val'][-1]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(self.model, self.opt, val_loss, epoch, history['info'])

        history['T'] = time.time() - t0
        history['time'] = sum(history['time'])

        print('   T:', history['T'])
        print('time:', history['time'])

        return history
