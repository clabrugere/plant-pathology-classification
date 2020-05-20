import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import cv2


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    '''Returns the output shape after a convolution, given input size and other convolution parameters.
    '''
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    
    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    
    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    '''Returns required padding given input size, convolution parameters and desired output size.
    '''
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)
    
    p_h = ((h_w_out[0] - 1)*stride[0] - h_w_in[0] + dilation[0]*(kernel_size[0]-1) + 1)
    p_w = ((h_w_out[1] - 1)*stride[1] - h_w_in[1] + dilation[1]*(kernel_size[1]-1) + 1)
    
    return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))


def mean_column_wise_roc_auc(y_hat, y):
    '''Compute the column averages AUC score (AUC mean over different classes).
    '''
    return roc_auc_score(y, y_hat, average='macro')


def plot_samples(ds):
    '''Plot some samples and their transformed version.
    '''
    n_samples = 6
    n_rows = 2
    n_cols = n_samples // n_rows 

    samples = np.random.choice(
        [i for i in range(len(ds))],
        size=n_samples,
        replace=False
    )
    ds.apply_transforms = False
    ds.to_tensor = False
    
    fig = plt.figure(figsize=(32, 8))
    outer = gridspec.GridSpec(n_rows, n_cols, wspace=0.2, hspace=0.2)
    
    for i in range(n_rows * n_cols):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        
        if ds.is_test:
            img_original = ds[samples[i]]
            img_transformed = ds._transform(img_original)
        else:
            img_original, label = ds[samples[i]]
            img_transformed = ds._transform(img_original)

        images = [img_original, img_transformed]

        for j in range(2):
            ax = plt.Subplot(fig, inner[j])
            ax.imshow(cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB))

            if not ds.is_test:
                ax.set_title(ds.label_from_vect(label))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
    
    ds.apply_transforms = True
    ds.to_tensor = True
    
    return fig