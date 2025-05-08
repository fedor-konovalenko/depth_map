## Intro

The main idea of the project is to find the best depth map prediction model for the inference on the mobile device. One of the main restrictions was the model's inference time, so the goal was to apply a kind of convolutional neural network, not visual transformer. The [RT Mono Depth model](https://arxiv.org/pdf/2308.10569v1) was chosen as a baseline and adopted for the project.

----

## Hypotheses

Compared to the basic model, an additional depth map was extracted from the encoder bottleneck and used as a depth distribution in the image.

---

## Datasets

- [NYU Depth V2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) (25% of the dataset) as a base dataset
- [ARKit Scenes](https://github.com/apple/ARKitScenes) Dataset as an additional dataset
---

## Training

The model was trained in Google Colab (A100). The standard PyTorch-based training loop was used.

The model training results.

<img src="https://drive.google.com/uc?export=view&id=1LheO4yYG6nL1gnlQ8qKI7Ji8WzvjSYla" width="600"/>

### Losses

The combination of [Structural Similarity Index (SSIM)](https://arxiv.org/pdf/2006.13846.pdf), L1, and [Smooth Loss](https://arxiv.org/pdf/1806.01260.pdf) was used for depth map.  
This loss function is based on L1 loss, while SSIM and Smooth losses help the model produce sharper details in the depth map.

$$
\text{Loss: } SSIM + L_1 + L_{\text{smooth}}
$$

$$
L = \alpha \cdot L_{\text{SSIM}} + (1 - \alpha) \cdot L_1 + \lambda \cdot L_{\text{smooth}}
$$

For depth distribution RMSE Loss was used.

### Metrics

For the  model quality assessment the following metrics were used
- ABSRel
- SQRel
- RMSE
- RMSE Logarithmic

$$AbsRel=\frac{1}{N}\sum \frac{|d_i - \widehat{d_i}|}{d_i}$$

$$SqRel=\frac{1}{N}\sum \frac{|d_i - \widehat{d_i}|^2}{d_i}$$

$$RMSE = \sqrt{\frac{1}{N}\sum |d_i - \widehat{d_i}|^2}$$

$$RMSE_{log} = \sqrt{\frac{1}{N}\sum |log_{10}d_i - log_{10}\widehat{d_i}|^2}$$

---

## Results

As a result, a model was trained and tested that provides the required inference time on a mobile device and has better quality compared to the base model. And smoothing the predicted depth maps using the moving average method, as shown in the [notebook](https://github.com/fedor-konovalenko/depth_map/blob/main/notebooks/tests.ipynb), allowed us to obtain higher-quality depth maps, which will allow us to use the model in applications using augmented reality.

[![Depth map example]("https://drive.google.com/uc?export=view&id=1ZwEi_482EEXbVE9bVaqGd-xZrsGpq1EY")]([https://drive.google.com/file/d/1aM30RxO1CObsautKWvDfDUWqh8aE1qjU/view])

---
