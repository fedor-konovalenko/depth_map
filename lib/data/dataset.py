import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.notebook import tqdm
import random
import csv


def read_fnames(csv_fname):
  with open(csv_fname) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    files = []
    for row in csv_reader:
      files.append(row)
  return files

def distribution(tensor, min_value=0, max_value=1, step=1.0/255):
    tensor_flat = tensor.flatten()
    bins = torch.arange(min_value, max_value + step, step)
    hist = torch.histc(tensor_flat, bins=bins.shape[0], min=min_value, max=max_value)
    return hist

def smooth(hist, device, steps=1):
  hist = hist.to(device)
  if steps == 1:
    return hist
  kernel_size = 3
  kernel = torch.ones(kernel_size) / kernel_size
  hist = hist.view(1, 1, -1).float()
  kernel = kernel.view(1, 1, -1).float()
  for i in range(steps):
    hist = F.conv1d(hist, kernel.to(device), padding=kernel_size//2)
  hist = hist.view(-1)
  return F.normalize(hist, dim=0)


stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

class MixDepth(Dataset):
  """
  Returns image with augmentations,
  depth map,
  flatten depth map scaled to meters (0-10)
  """
  def __init__(self, fnames, device, mode='train', size=(256, 256), inmemory=False):
    super().__init__()
    self.inmemory = inmemory
    self.fnames = fnames
    self.size = size
    self.mode = mode
    self.device = device

    flipper = random.random()

    rgb_transforms = [T.ToTensor(),
                      T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
                      T.Normalize(*stats)]
    depth_transforms = [T.ToTensor()]


    if self.mode == 'train':
        if flipper > 0.5:
            rgb_transforms.append(T.RandomHorizontalFlip(1.0))
            depth_transforms.append(T.RandomHorizontalFlip(1.0))

    self.rgb_transform = T.Compose(rgb_transforms)
    self.depth_transform = T.Compose(depth_transforms)

    if inmemory:
      self.rgbs = []
      self.depths = []
      self.distrs = []

      gpu_total_memory = torch.cuda.get_device_properties(0).total_memory
      max_gpu_usage = gpu_total_memory - 5 * (1024 ** 3)
      memory_warning_was_shown = False

      for fnames_ in tqdm(fnames, desc="Load data to GPU memory"):
        rgb, depth, distr = self.read_pair(fnames_)
        rgb, depth, distr = rgb.half(), depth.half(), distr.half()
        if torch.cuda.memory_allocated(0) < max_gpu_usage:
          rgb, depth, distr = rgb.to(device), depth.to(device), distr.to(device)
        elif not memory_warning_was_shown:
          memory_warning_was_shown = True
          print("Limit of GPU Memory. Ram is used")
        self.rgbs.append(rgb)
        self.depths.append(depth)
        self.distrs.append(distr)

  def read_img(self, fname, transform):
    img = Image.open(fname)
    img = img.resize(self.size, resample=Image.BILINEAR)
    img = transform(img)
    return img

  def read_pair(self, fnames):
    rgb = self.read_img(fnames[0], self.rgb_transform)
    depth = self.read_img(fnames[1], self.depth_transform)
    return rgb, depth, smooth(distribution(depth), self.device,4)

  def __len__(self):
    return len(self.fnames)

  def __getitem__(self, idx):
    if self.inmemory:
      rgb, depth, distr = self.rgbs[idx].float(), self.depths[idx].float(), self.distrs[idx].float()
    else:
      rgb, depth, distr = self.read_pair(self.fnames[idx])
    return rgb, depth, distr