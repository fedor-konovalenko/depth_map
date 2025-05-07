import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
  """
  https://github.com/nianticlabs/monodepth2/blob/master/layers.py
  """
  def __init__(self):
    super(SSIMLoss, self).__init__()
    self.mu_x_pool   = nn.AvgPool2d(3, 1)
    self.mu_y_pool   = nn.AvgPool2d(3, 1)
    self.sig_x_pool  = nn.AvgPool2d(3, 1)
    self.sig_y_pool  = nn.AvgPool2d(3, 1)
    self.sig_xy_pool = nn.AvgPool2d(3, 1)

    self.refl = nn.ReflectionPad2d(1)

    self.C1 = 0.01 ** 2
    self.C2 = 0.03 ** 2

  def forward(self, x, y):
    x = self.refl(x)
    y = self.refl(y)

    mu_x = self.mu_x_pool(x)
    mu_y = self.mu_y_pool(y)

    sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
    sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
    sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

    loss = (1 - SSIM_n / SSIM_d) / 2
    loss = torch.clamp(loss, 0, 1)
    return loss

class SmoothLoss(nn.Module):
  """
  https://github.com/nianticlabs/monodepth2/blob/master/layers.py
  """
  def __init__(self):
    super(SmoothLoss, self).__init__()

  def forward(self, disp, img):
      grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
      grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

      grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
      grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

      grad_disp_x *= torch.exp(-grad_img_x)
      grad_disp_y *= torch.exp(-grad_img_y)

      loss = grad_disp_x.mean() + grad_disp_y.mean()
      return loss

class PELoss(nn.Module):
  def __init__(self, a=0.85, l=0.05):
    super(PELoss, self).__init__()
    self.a = a
    self.l = l
    self.ssim = SSIMLoss()
    self.smooth = SmoothLoss()

  def forward(self, target, pred, rgb):
    abs_diff = torch.abs(target - pred)
    pe = self.a * self.ssim(pred, target) + (1 - self.a) * abs_diff
    pe = pe.mean()
    pe += self.l * self.smooth(pred, rgb)
    return pe

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def multiscale_loss(criterion_map, criterion_distr, distr, target, D0, D1, D2, D3, DD, rgb, d_weight):
  def scale_loss(PRED):
    size = PRED.shape[-2:]
    GT =  F.interpolate(target, size=size, mode='bilinear')
    RGB = F.interpolate(rgb,    size=size, mode='bilinear')
    return criterion_map(GT, PRED, RGB)

  loss0 = criterion_map(target, D0, rgb)
  loss1 = scale_loss(D1)
  loss2 = scale_loss(D2)
  loss3 = scale_loss(D3)
  loss_d = criterion_distr(DD, distr)
  loss = (loss0 + loss1 + loss2 + loss3) * (1 - d_weight) / 4 + loss_d * d_weight

  return loss