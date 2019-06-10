"""
Adapted from Michael Gygli's snippets: 
https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
"""
import io
import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def log_image(thumbs):
    
    if thumbs.dtype == np.float:
        thumbs = (255*thumbs).astype(np.uint8)
    
    s = io.BytesIO()
    imageio.imsave(s, thumbs, format='png')
    return tf.Summary.Image(encoded_image_string=s.getvalue(), height=thumbs.shape[0], width=thumbs.shape[1])


def log_plot(fig):
    s = io.BytesIO()
    fig.savefig(s, format='png', bbox_inches='tight')
    plt.close(fig)
    return imageio.imread(s.getvalue(), pilmode='RGB')


def log_histogram(values, bins=50):
    # Create histogram using numpy        
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)
        
    return hist
