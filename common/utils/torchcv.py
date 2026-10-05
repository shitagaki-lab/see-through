import torch
from einops import reduce
import numpy as np
import cv2

from utils.depth_sort import median_depth


def build_alpha_pyramid_torch(argb_tensor, dk=1.2):
    argb_tensor = argb_tensor.clone()
    pyramid = []
    argb_tensor[:, 1:] = argb_tensor[:, 1:] * argb_tensor[:, [0]]

    while True:
        pyramid.append(argb_tensor)

        B, C, H, W = argb_tensor.shape
        if min(H, W) == 1:
            break

        argb_tensor = torch.nn.functional.interpolate(argb_tensor, (int(W / dk), int(H / dk)), align_corners=False, mode='bilinear')

    return pyramid[::-1]


def pad_rgb_torch(argb_tensor, return_format='rgb', input_format='argb'):
    '''
    argb_tensor: (b c h w) normalized to 0~1
    '''
    # Written by lvmin at Stanford
    # Massive iterative Gaussian filters are mathematically consistent to pyramid.

    if input_format == 'rgba':
        argb_tensor = torch.cat([argb_tensor[:, [3]], argb_tensor[:, :3]], dim=1)


    alpha = argb_tensor[:, :1]
    pyramid = build_alpha_pyramid_torch(argb_tensor)

    fg = pyramid[0]
    # fg = np.sum(top_c, axis=(0, 1), keepdims=True) / np.sum(top_a, axis=(0, 1), keepdims=True).clip(1e-8, 1e32)
    fg = reduce(fg[:, 1:], 'b c h w -> b c 1 1', 'sum') / reduce(fg[:, [0]], 'b c h w -> b c 1 1', 'sum').clip(1e-8, 1e32)

    for argb_tensor in pyramid:
        layer_c = argb_tensor[:, 1:]
        layer_a = argb_tensor[:, :1]
        b, c, layer_h, layer_w = layer_c.shape
        fg = torch.nn.functional.interpolate(fg, (layer_w, layer_h), align_corners=False, mode='bilinear')
        # fg = cv2.resize(fg, (layer_w, layer_h), interpolation=cv2.INTER_LINEAR)
        fg = layer_c + fg * (1.0 - layer_a)

    if return_format == 'argb':
        fg = torch.cat([alpha, fg], axis=1)

    return fg


def cluster_inpaint_part(depth, mask, img, inpaint='lama',**kwargs):
    from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, KMeans
    import random

    rgb = img[..., :3]
    alpha = img[..., -1]

    dmin, dmax = depth[mask].min(), depth[mask].max()
    d = ((depth - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)
    h, w = d.shape[:2]

    # d_blured = cv2.medianBlur(d, ksize=3)
    dsamples = samples = d[mask] / 255.

    max_nsamples = 2000
    
    if len(d) > max_nsamples:
        shuffle_ids = list(range(len(d)))
        random.shuffle(shuffle_ids)
        samples = samples[shuffle_ids[:max_nsamples]]
    cluster = KMeans(n_clusters=2, n_init="auto").fit(samples[:, None])

    labels = np.full((h, w), fill_value=-1)
    labels[mask] = cluster.predict(dsamples[:, None])

    i2c = np.argsort(cluster.cluster_centers_.flatten())
    extracted_parts = []

    if inpaint == 'lama':
        from annotators.lama_inpainter import apply_inpaint
        from utils.io_utils import save_tmp_img
        inpaint_method = apply_inpaint
    else:
        inpaint_method = lambda img, mask, *args, **kwargs: cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    for ii in range(len(cluster.cluster_centers_) - 1):
        to_mask = labels == i2c[ii]
        imask = to_mask.astype(np.uint8) * 255

        ksize = 3
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
        imask_inpaint = cv2.dilate(imask, element)

        sigma = 0.7
        imask = cv2.GaussianBlur(imask, [3] * 2, sigmaX=sigma, sigmaY=sigma)

        extracted_parts.append({
            'img': np.concatenate([rgb, imask[..., None]], axis=-1),
            # 'depth_median': cluster.cluster_centers_[ii] * (dmax - dmin + 1e-6) + dmin
            # a cluster can end up with no pixels at all -> np.median([]) is NaN
            'depth_median': median_depth(depth[to_mask]),
            'depth': d.astype(np.float32) / 255. * (dmax - dmin + 1e-6) + dmin
        })

        valid_mask = np.clip((alpha.astype(np.int32) - imask.astype(np.int32)), 0, 255) > 50
        if inpaint == 'lama':
            a = alpha[..., None] / 255.

            rgb_values = np.mean(rgb[valid_mask])
            if rgb_values < 100:
                fill = np.array([255] * 3)
            else:
                fill = np.array([0] * 3)
            rgb = np.round(rgb * a + (1-a) * fill).astype(np.uint8)

        rgb = inpaint_method(rgb, imask_inpaint)

        if inpaint == 'lama':
            dist_map = np.mean(np.abs(rgb.astype(np.float32) - fill[None, None]), axis=2)
            # save_tmp_img(np.clip((alpha.astype(np.int32) - imask.astype(np.int32)), 0, 255).astype(np.uint8))
            
            m = dist_map > 15
            dist_map[m] = 255
            dist_map[np.bitwise_not(m)] = dist_map[np.bitwise_not(m)]

            dist_map[imask_inpaint <= 127] = alpha[imask_inpaint <=127]
            alpha = np.round(dist_map).astype(np.uint8)
        else:
            alpha = inpaint_method(alpha, imask)
        # d = inpaint_method(d, imask)
        d[imask > 127] = np.median(d[valid_mask])

    to_mask = labels == i2c[-1]
    extracted_parts.append({'img': np.concatenate([rgb, alpha[..., None]], axis=-1), 
        'depth_median': median_depth(depth[to_mask]),
        'depth': d.astype(np.float32) / 255. * (dmax - dmin + 1e-6) + dmin
        }
    )
    
    if 'xyxy' in kwargs:
        for d in extracted_parts:
            d['xyxy'] = kwargs['xyxy']
    
    layer_name = None
    if 'tag' in kwargs:
        layer_name = kwargs['tag']
    if 'layer_name' in kwargs:
        layer_name = kwargs['layer_name']

    # if layer_name is not None:
    #     for ii, d in enumerate(extracted_parts):
    #         d['layer_name'] = layer_name + f'_{ii}'
    #         d['tag'] = layer_name + f'_{ii}'

    return extracted_parts
