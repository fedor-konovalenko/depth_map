import torch
import matplotlib.pyplot as plt


def torch_to_img(x):
    x = torch.moveaxis(x, x.ndim - 3, x.ndim - 1)
    x = (x + 1) * 127.5
    x = x.type(torch.uint8)
    x = torch.clamp(x, 0, 255)
    x = x.detach().cpu().numpy()
    return x


def _plot_img(cnt, img, title, i, cmap='gnuplot2'):
    plt.subplot(3, 6, cnt + 3 + 6 * i)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')


def _plot_hist(item, label, i, yscale='linear'):
    plt.subplot(2, 3, i)
    plt.plot(item['train'], label='train: %.4f' % (item['train'][-1]))
    plt.plot(item['val'], label='   val: %.4f' % (item['val'][-1]))
    plt.legend(loc='best')
    plt.yscale(yscale)
    plt.ylabel(label)


def _plot_score(item, label, i, yscale='linear'):
    plt.subplot(4, 6, i)
    plt.plot(item['train'], label='train: %.4f' % (item['train'][-1]))
    plt.plot(item['val'], label='  val: %.4f' % (item['val'][-1]))
    plt.legend(loc='best', title=label, fontsize='x-small', title_fontsize='x-small')
    plt.yscale(yscale)
    plt.xticks([])
    plt.yticks(fontsize=6)
