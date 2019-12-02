import numpy as np
import torch


def truncated_normal(uniform):
    return parameterized_truncated_normal(uniform, mu=0.0, sigma=1.0, a=-2, b=2)


def sample_truncated_normal(shape=()):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)))


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    r"""Implements the sampling of truncated normal distribution using the inversed
        cumulative distribution function (CDF) method.

    .. _Truncated Normal\: normal distribution in which the range of definition is
                           made finite at one or both ends of the interval.
        https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    normal = torch.distributions.normal.Normal(0, 1)

    alpha, beta = (a - mu) / sigma, (b - mu) / sigma
    p = normal.cdf(alpha) + (normal.cdf(beta) - normal.cdf(alpha)) * uniform
    p = p.numpy()

    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x.type(torch.get_default_dtype())


def partCombiner(eyel, eyer, nose, mouth, IMAGE_SIZE=128):
    """
       x          y
       43.5823    41.0000
       86.4177    41.0000
       64.1165    64.7510
       47.5863    88.8635
       82.5904    89.1124
       this is the mean location of 5 landmarks.
    """
    EYE_H, EYE_W = eyel.shape[2:]
    NOSE_H, NOSE_W = nose.shape[2:]
    MOUTH_H, MOUTH_W = mouth.shape[2:]

    eyel_p = torch.nn.functional.pad(eyel, (int(44-EYE_W/2-1), int(IMAGE_SIZE-(44+EYE_W/2-1)),
                                            int(41-EYE_H/2-1), int(IMAGE_SIZE-(41+EYE_H/2-1))))
    eyer_p = torch.nn.functional.pad(eyer, (int(86-EYE_W/2-1), int(IMAGE_SIZE-(86+EYE_W/2-1)),
                                            int(41-EYE_H/2-1), int(IMAGE_SIZE-(41+EYE_H/2-1))))
    nose_p = torch.nn.functional.pad(nose, (int(64-NOSE_W/2-1), int(IMAGE_SIZE-(64+NOSE_W/2-1)),
                                            int(65-NOSE_H/2-1), int(IMAGE_SIZE-(65+NOSE_H/2-1))))
    mouth_p = torch.nn.functional.pad(mouth, (int(65-MOUTH_W/2-1), int(IMAGE_SIZE-(65+MOUTH_W/2-1)),
                                              int(89-MOUTH_H/2-1), int(IMAGE_SIZE-(89+MOUTH_H/2-1))))

    return torch.max(torch.stack([eyel_p, eyer_p, nose_p, mouth_p], dim=1), dim=1)[0]


def symL1(images):
    """Symmetry Loss. """
    assert images.ndim == 4 and all(images.shape)

    _l = images.size(3) // 2
    _r = images.size(3) - _l

    left = images[..., :_l].data
    right = images[..., _r:]

    return (left - right.flip(dims=(3,))).abs().mean(dim=(2, 3))


def total_varation(images):
    r"""Calculate and return the Total Variation for one or more images.

    .. _Total Variation\: the sum of the absolute differences for neighboring pixel-values
                          in the input images.
        https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    if images.ndim == 3:
        images = images.unsqueeze(dim=0)

    assert images.ndim == 4 and all(images.shape), "images must be of size [C, H, W] or [N, C, H, W]"

    pixel_dif1 = images[..., 1:, :] - images[..., :-1, :]
    pixel_dif2 = images[..., 1:] - images[..., :-1]

    tot_var = pixel_dif1.abs().sum(dim=(1, 2, 3)) + pixel_dif2.abs().sum(dim=(1, 2, 3))

    return tot_var
