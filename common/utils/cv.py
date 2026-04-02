import math
from typing import Any, List, Union, Tuple, Dict, Optional
import os
import random
from enum import Enum

import cv2
import numpy as np
import pycocotools.mask as maskUtils
from PIL import Image


from .io_utils import json2dict, dict2json


def bbox_intersection(xyxy, xyxy2):
    x1, y1, x2, y2 = xyxy2
    dx1, dy1, dx2, dy2 = xyxy
    ix1, ix2 = max(x1, dx1), min(x2, dx2)
    iy1, iy2 = max(y1, dy1), min(y2, dy2)
    if ix2 >= ix1 and iy2 >= iy1:
        return [ix1, iy1, ix2, iy2]
    return None


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return (codebook[labels].reshape(w, h, -1) * 255).astype(np.uint8)


def quantize_image(image: np.ndarray, n_colors: int, method='kmeans', mask=None):

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin
    from sklearn.utils import shuffle

    # https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
    image = np.array(image, dtype=np.float64) / 255

    if len(image.shape) == 3:
        w, h, d = tuple(image.shape)
    else:
        w, h = image.shape
        d = 1

    # assert d == 3
    image_array = image.reshape(-1, d)

    if method == 'kmeans':

        image_array_sample = None
        if mask is not None and not np.all(mask):
            ids  = np.where(mask)
            if len(ids[0]) > 10:
                bg = image[ids][::2]
                fg = image[np.where(mask == 0)]
                max_bg_num = int(fg.shape[0] * 1.5)
                if bg.shape[0] > max_bg_num:
                    bg = shuffle(bg, random_state=0, n_samples=max_bg_num)
                image_array_sample = np.concatenate((fg, bg), axis=0)
                if image_array_sample.shape[0] > 2048:
                    image_array_sample = shuffle(image_array_sample, random_state=0, n_samples=2048)
                else:
                    image_array_sample = None

        if image_array_sample is None:
            image_array_sample = shuffle(image_array, random_state=0, n_samples=2048)
        
        kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=0).fit(
            image_array_sample
        )

        labels = kmeans.predict(image_array)
        quantized  = recreate_image(kmeans.cluster_centers_, labels, w, h)
        return quantized, kmeans.cluster_centers_, labels

    else:

        codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
        labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)

        return [recreate_image(codebook_random, labels_random, w, h)]



def get_template_histvq(template: np.ndarray) -> Tuple[List[np.ndarray]]:
    len_shape = len(template.shape)
    num_c = 3
    mask = None
    if len_shape == 2:
        num_c = 1
    elif len_shape == 3 and template.shape[-1] == 4:
        mask = np.where(template[..., -1])
        template = template[..., :num_c][mask]

    values, quantiles = [], []
    for ii in range(num_c):
        v, c = np.unique(template[..., ii].ravel(), return_counts=True)
        q = np.cumsum(c).astype(np.float64)
        if len(q) < 1:
            return None, None
        q /= q[-1]
        values.append(v)
        quantiles.append(q)
    return values, quantiles


def inplace_hist_matching(img: np.ndarray, tv: List[np.ndarray], tq: List[np.ndarray]) -> None:
    len_shape = len(img.shape)
    num_c = 3
    mask = None

    tgtimg = img
    if len_shape == 2:
        num_c = 1
    elif len_shape == 3 and img.shape[-1] == 4:
        mask = np.where(img[..., -1])
        tgtimg = img[..., :num_c][mask]

    im_h, im_w = img.shape[:2]
    oldtype = img.dtype
    for ii in range(num_c):
        _, bin_idx, s_counts = np.unique(tgtimg[..., ii].ravel(), return_inverse=True,
                                                return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        if len(s_quantiles) == 0:
            return
        s_quantiles /= s_quantiles[-1]
        interp_t_values = np.interp(s_quantiles, tq[ii], tv[ii]).astype(oldtype)
        if mask is not None:
            img[..., ii][mask] = interp_t_values[bin_idx]
        else:
            img[..., ii] = interp_t_values[bin_idx].reshape((im_h, im_w))
            # try:
            #     img[..., ii] = interp_t_values[bin_idx].reshape((im_h, im_w))
            # except:
            #     LOGGER.error('##################### sth goes wrong')
            #     cv2.imshow('img', img)
            #     cv2.waitKey(0)


def fgbg_hist_matching(fg_list: List, bg: np.ndarray, min_tq_num=128, fg_only=False):
    '''
    inplace op
    '''
    btv, btq = get_template_histvq(bg)
    ftv, ftq = get_template_histvq(fg_list[0])
    num_fg = len(fg_list)
    idx_matched = -1
    if num_fg > 1:
        _ftv, _ftq = get_template_histvq(fg_list[0])
        if _ftq is not None and ftq is not None:
            if len(_ftq[0]) > len(ftq[0]):
                idx_matched = num_fg - 1
                ftv, ftq = _ftv, _ftq
            else:
                idx_matched = 0

    if fg_only and ftq is not None:
        tv, tq = ftv, ftq
        if len(tq[0]) > min_tq_num:
            inplace_hist_matching(bg, tv, tq)
        return

    if btq is not None and ftq is not None:
        if len(btq[0]) > len(ftq[0]):
            tv, tq = btv, btq
            idx_matched = -1
        else:
            tv, tq = ftv, ftq
            if len(tq[0]) > min_tq_num:
                inplace_hist_matching(bg, tv, tq)
        
        if len(tq[0]) > min_tq_num:
            for ii, fg in enumerate(fg_list):
                if ii != idx_matched and len(tq[0]) > min_tq_num:
                    inplace_hist_matching(fg, tv, tq)


def mask2rle(mask: np.ndarray, decode_for_json: bool = True) -> Dict:
    assert mask.ndim == 2
    mask_rle = maskUtils.encode(np.array(
                        mask[..., np.newaxis] > 0, order='F',
                        dtype='uint8'))[0]
    if decode_for_json:
        mask_rle['counts'] = mask_rle['counts'].decode()
    return mask_rle


def rle2mask(rle: Union[Dict, str], to_bool=True):
    # if isinstance(rle, Dict):
    #     rle = rle['counts']
    mask = maskUtils.decode(rle)
    if to_bool:
        return mask > 0
    return mask

def batch_save_masks(masks: np.ndarray, savep: str, compress=None, mask_meta_list=None):
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            masks = masks[None]
    masks = [mask2rle(m) for m in masks]
    if mask_meta_list is not None:
        assert len(masks) == len(mask_meta_list)
        for m, meta in zip(masks, mask_meta_list):
            m.update(meta)
    dict2json(masks, savep, compress=compress)


def batch_load_masks(p: str, to_bool=True):
    masks = json2dict(p)
    masks = [rle2mask(m, to_bool=to_bool) for m in masks]

    return masks


def smart_resize(src: np.ndarray, target_size, upscale_interpolation=cv2.INTER_LINEAR, downscale_interpolation=cv2.INTER_AREA):
    h, w = src.shape[:2]
    th, tw = target_size
    if th == h and tw == w:
        return src.copy()
    if th * tw < h * w:
        interpolation = downscale_interpolation
    else:
        interpolation = upscale_interpolation
    return cv2.resize(src, (tw, th), interpolation=interpolation)



def validate_resolution(resolution: Union[str, Tuple, int], div=-1) -> List:

    '''
    make sure resolution is a valid (h: int, w: int) format and can be divided by div
    '''

    if isinstance(resolution, str):
        resolution = [int(r.strip()) for r in resolution.split(',')]
    elif isinstance(resolution, str):
        resolution = (resolution, resolution)
    elif isinstance(resolution, int):
        resolution = [resolution, resolution]
    elif isinstance(resolution, (Tuple, List)):
        resolution = list(resolution)[:2]

    reso_out = []
    for res in resolution:
        if div > 0 and res % div != 0:
            res = math.ceil(res / div) * div
        reso_out.append(res)
    
    return reso_out


def center_square_pad_resize(img: np.ndarray, target_size, pad_value=0, upscale_interpolation=cv2.INTER_LINEAR, downscale_interpolation=cv2.INTER_AREA, return_pad_info=False):
    h, w = img.shape[:2]
    pad_size = (w, h)
    pad_pos = (0, 0)
    if h != w:
        sz = max(h, w)
        px1 = (sz - w) // 2
        py1 = (sz - h) // 2
        shape = (sz, sz) if img.ndim == 2 else (sz, sz, img.shape[-1])
        padded = np.full(shape, pad_value, dtype=img.dtype)
        padded[py1: py1 + h, px1: px1 + w] = img
        h, w = padded.shape[:2]
        img = padded
        pad_size = (w, h)
        pad_pos = (px1, py1)
    if h != target_size or w != target_size:
        img = smart_resize(img, (target_size, target_size), upscale_interpolation=upscale_interpolation, downscale_interpolation=downscale_interpolation)
    if return_pad_info:
        return img, pad_size, pad_pos
    else:
        return img


def random_hsv(img, hgain: float = 0.015, sgain: float = 0.6, vgain: float = 0.4):
    if hgain or sgain or vgain:
        dtype = img.dtype  # uint8

        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
        x = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * (r[0] + 1)) % 180).astype(dtype)   # original hue implementation from ultralytics<=8.3.78
        lut_hue = ((x + r[0] * 180) % 180).astype(dtype)
        lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(dtype)
        lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)
        lut_sat[0] = 0  # prevent pure white changing color, introduced in 8.3.79

        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=img)  # no return needed

    return img


def resize_short_side_to(src: np.ndarray, short_side: int):
    h, w = src.shape[:2]
    th, tw = h, w
    if h > w:
        tw = short_side
        th = int(round(h / w * short_side))
    else:
        th = short_side
        tw = int(round(w / h * short_side))
    return smart_resize(src, (th, tw))


def random_crop(src: np.ndarray, target_size) -> None:
    '''
    target_size: (h, w)
    '''
    h, w = src.shape[:2]
    th, tw = target_size

    scale = max(th / h, tw / w)
    if scale > 1:
        h = math.ceil(h * scale)
        w = math.ceil(w * scale)
        src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    
    x0 = y0 = 0
    if h > th:
        y0 = random.randint(0, h - th)
    if w > tw:
        x0 = random.randint(0, w - tw)

    return src[y0: y0 + th, x0: x0 + tw].copy()


def img_bbox(src: np.ndarray):
    if isinstance(src, Image.Image):
        src = np.array(src)
    if src.ndim == 3 and src.shape[-1] == 4:
        src = src[..., 0]
    return cv2.boundingRect(cv2.findNonZero(src.astype(np.uint8)))


def mask_xyxy(mask):
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    bbox = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
    x1, y1, x2, y2 = bbox
    x2 += x1
    y2 += y1
    return [x1, y1, x2, y2]


def argb2rgba(argb):
    return np.concatenate([argb[..., 1:], argb[..., [0]]], axis=2)


def img_alpha_blending(
    drawables: List[np.ndarray], 
    xyxy=None, 
    output_type='numpy', 
    final_size=None, 
    max_depth_val=255,
    premultiplied=True,
):
    '''
    final_size: (h, w)
    '''

    if isinstance(drawables, (np.ndarray, dict)):
        drawables = [drawables]

    # infer final scene size
    if xyxy is not None:
        final_size = [xyxy[3] - xyxy[1], xyxy[2] - xyxy[0]]
        x1, y1, x2, y2 = xyxy
    elif final_size is None:
        d = drawables[0]
        if isinstance(d, dict):
            d = d['img']
        final_size = d.shape[:2]

    final_rgb = np.zeros((final_size[0], final_size[1], 3), dtype=np.float32)
    final_alpha = np.zeros_like(final_rgb[..., [0]])
    final_depth = None

    for drawable_img in drawables:
        dxyxy = None
        depth = None
        if isinstance(drawable_img, dict):
            depth = drawable_img.get('depth', None)
            tag = drawable_img.get('tag', None)
            if depth is not None:
                if depth.ndim == 2:
                    depth = depth[..., None]
                if final_depth is None:
                    final_depth = np.full_like(final_alpha, fill_value=max_depth_val)
            if 'xyxy' in drawable_img:
                dxyxy = drawable_img['xyxy']
                dx1, dy1, dx2, dy2 = dxyxy
            drawable_img = drawable_img['img']
            if dxyxy is not None:
                if dx1 < 0:
                    drawable_img = drawable_img[:, -dx1:]
                    if depth is not None:
                        depth = depth[:, -dx1:]
                    dx1 = 0
                if dy1 < 0:
                    drawable_img = drawable_img[-dy1:]
                    if depth is not None:
                        depth = depth[-dy1:]
                    dy1 = 0

        if drawable_img.ndim == 3 and drawable_img.shape[-1] == 3:
            drawable_alpha = np.ones_like(drawable_img[..., [-1]])
        else:
            drawable_alpha = drawable_img[..., [-1]] / 255

        drawable_img = drawable_img[..., :3]

        if xyxy is not None:
            if dxyxy is None:
                drawable_img = drawable_img[y1: y2, x1: x2]
            else:
                intersection = bbox_intersection(xyxy, dxyxy)
                if intersection is None:
                    continue
                ix1, iy1, ix2, iy2 = intersection
                if depth is not None:
                    depth = depth[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                    drawable_alpha = drawable_alpha[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                    update_mask = (final_depth[iy1-y1: iy2-y1, ix1-x1: ix2-x1] > depth).astype(np.uint8)
                    final_depth[iy1-y1: iy2-y1, ix1-x1: ix2-x1] = update_mask * depth + (1-update_mask) * final_depth[iy1-y1: iy2-y1, ix1-x1: ix2-x1]
                    drawable_img = drawable_img[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                    final_rgb[iy1-y1: iy2-y1, ix1-x1: ix2-x1] = update_mask * (final_rgb[iy1-y1: iy2-y1, ix1-x1: ix2-x1] * (1-drawable_alpha) + drawable_img) + \
                        (1 - update_mask) * (drawable_img * (1-final_alpha[iy1-y1: iy2-y1, ix1-x1: ix2-x1]) + final_rgb[iy1-y1: iy2-y1, ix1-x1: ix2-x1])
                    final_alpha[iy1-y1: iy2-y1, ix1-x1: ix2-x1] = np.clip(final_alpha[iy1-y1: iy2-y1, ix1-x1: ix2-x1] + drawable_alpha, 0, 1)
                else:
                    
                    drawable_alpha = drawable_alpha[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                    final_alpha[iy1-y1: iy2-y1, ix1-x1: ix2-x1] += drawable_alpha
                    drawable_img = drawable_img[iy1-dy1: iy2-dy1, ix1-dx1: ix2-dx1]
                    final_rgb[iy1-y1: iy2-y1, ix1-x1: ix2-x1] = final_rgb[iy1-y1: iy2-y1, ix1-x1: ix2-x1] * (1-drawable_alpha) + drawable_img
                    continue

        elif dxyxy is None:
            if depth is not None:
                update_mask = (final_depth > depth).astype(np.uint8)
                final_depth = update_mask * depth + (1-update_mask) * final_depth
                final_rgb = update_mask * (final_rgb * (1-drawable_alpha) + drawable_img) + \
                    (1 - update_mask) * (drawable_img * (1-final_alpha) + final_rgb)
                final_alpha = np.clip(final_alpha + drawable_alpha, 0, 1)
            else:
                final_alpha += drawable_alpha
                final_alpha = np.clip(final_alpha, 0, 1)
                if not premultiplied:
                    drawable_img = drawable_img * drawable_alpha
                final_rgb = final_rgb * (1 - drawable_alpha) + drawable_img
        else:
            if depth is not None:
                update_mask = (final_depth[dy1: dy2, dx1: dx2] > depth).astype(np.uint8)
                update_mask = update_mask * (drawable_alpha > 0.1)
                final_depth[dy1: dy2, dx1: dx2] = update_mask * depth + (1-update_mask) * final_depth[dy1: dy2, dx1: dx2]
                final_rgb[dy1: dy2, dx1: dx2] = update_mask * (final_rgb[dy1: dy2, dx1: dx2] * (1-drawable_alpha) + drawable_img) + \
                    (1 - update_mask) * (drawable_img * (1-final_alpha[dy1: dy2, dx1: dx2]) + final_rgb[dy1: dy2, dx1: dx2])
                final_alpha[dy1: dy2, dx1: dx2] = np.clip(final_alpha[dy1: dy2, dx1: dx2] + drawable_alpha, 0, 1)
            else:
                final_alpha[dy1: dy2, dx1: dx2] += drawable_alpha
                final_alpha = np.clip(final_alpha, 0, 1)
                if not premultiplied:
                    drawable_img = drawable_img * drawable_alpha
                final_rgb[dy1: dy2, dx1: dx2] = final_rgb[dy1: dy2, dx1: dx2] * (1-drawable_alpha) + drawable_img

    final_alpha = np.clip(final_alpha, 0, 1) * 255
    final = np.concatenate([final_rgb, final_alpha], axis=2)
    final = np.clip(final, 0, 255).astype(np.uint8)

    output_type = output_type.lower()
    if output_type == 'pil':
        final = Image.fromarray(final)
    elif output_type == 'dict':
        final = {
            'img': final
        }
        if final_depth is not None:
            final['depth'] = final_depth

    return final


def rgba_to_rgb_fixbg(img: np.ndarray, background_color=255):
    if isinstance(img, Image.Image):
        img = np.array(img)
    assert img.ndim == 3
    if img.shape[-1] == 3:
        return img
    if isinstance(background_color, int):
        bg = np.full_like(img[..., :3], fill_value=background_color)
    else:
        background_color = np.array(background_color)[:3].astype(np.uint8)
        bg = np.full_like(img[..., :3], fill_value=255)
        bg[..., :3] = background_color
    return img_alpha_blending([bg, img])[..., :3].copy()


def build_alpha_pyramid(color, alpha, dk=1.2):
    # Written by lvmin at Stanford
    # Massive iterative Gaussian filters are mathematically consistent to pyramid.

    pyramid = []
    current_premultiplied_color = color * alpha
    current_alpha = alpha

    while True:
        pyramid.append((current_premultiplied_color, current_alpha))

        H, W, C = current_alpha.shape
        if min(H, W) == 1:
            break

        current_premultiplied_color = cv2.resize(current_premultiplied_color, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)
        current_alpha = cv2.resize(current_alpha, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)[:, :, None]
    return pyramid[::-1]


def pad_rgb(np_rgba_hwc_uint8, return_format='rgb', to_uint8=False, keep_ori_pixel=True):
    # Written by lvmin at Stanford
    # Massive iterative Gaussian filters are mathematically consistent to pyramid.


    np_rgba_hwc = np_rgba_hwc_uint8.astype(np.float32) / 255.0

    if keep_ori_pixel:
        ori_rgb = np_rgba_hwc[..., :3].copy()
        ori_alpha = np_rgba_hwc[..., [-1]].copy()

    pyramid = build_alpha_pyramid(color=np_rgba_hwc[..., :3], alpha=np_rgba_hwc[..., 3:])

    top_c, top_a = pyramid[0]
    fg = np.sum(top_c, axis=(0, 1), keepdims=True) / np.sum(top_a, axis=(0, 1), keepdims=True).clip(1e-8, 1e32)

    for layer_c, layer_a in pyramid:
        layer_h, layer_w, _ = layer_c.shape
        fg = cv2.resize(fg, (layer_w, layer_h), interpolation=cv2.INTER_LINEAR)
        fg = layer_c + fg * (1.0 - layer_a)

    if keep_ori_pixel:
        fg = np.clip(ori_alpha * ori_rgb + (1-ori_alpha) * fg, 0, 1)

    if return_format == 'argb':
        fg = np.concatenate([np_rgba_hwc[..., 3:], fg], axis=2)

    if to_uint8:
        fg = (fg * 255).astype(np.uint8)

    return fg


def checkerboard_vis(img):
    # y = y.clip(0, 1).movedim(1, -1)
    # alpha = y[..., :1]
    # fg = y[..., 1:]
    H, W, C = img.shape
    img = img.astype(np.float32) / 255.
    alpha = img[..., [-1]]
    fg = img[..., :3]
    cb = checkerboard(shape=(H // 64, W // 64))
    cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
    cb = (0.5 + (cb - 0.5) * 0.1)[..., None]
    # cb = torch.from_numpy(cb).to(fg)

    vis = fg * alpha + cb * (1 - alpha)
    vis = (vis * 255.0).clip(0, 255).astype(np.uint8)
    return vis


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


def visualize_rgba(rgba):
    rgba = rgba.astype(np.float32) / 255.
    H, W, C = rgba.shape
    cb = checkerboard(shape=(H // 64, W // 64))
    cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST)
    cb = (0.5 + (cb - 0.5) * 0.1)[..., None]
    alpha = rgba[..., [-1]]
    fg = rgba[..., :3]
    vis = (fg * alpha + cb * (1 - alpha))
    vis = np.clip(vis * 255.0, 0, 255).astype(np.uint8)
    return vis


class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_rectangle_mask(shape, margin=10, bbox_min_size=64, bbox_max_size=384, min_times=1, max_times=4):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask


def make_random_irregular_mask(shape, max_angle=4, max_len=600, max_width=256, min_times=1, max_times=5,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 10 + np.random.randint(max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask


def random_pad_img(img: np.array, tmax=0, bmax=0, lmax=0, rmax=0, pad_values=0):
    l = r = t = b = 0
    if tmax > 0:
        t = random.randint(0, tmax)
    if bmax > 0:
        b = random.randint(0, bmax)
    if lmax > 0:
        l = random.randint(0, lmax)
    if rmax > 0:
        r = random.randint(0, rmax)
    if t > 0 or b > 0 or l > 0 or r > 0:
        padded = cv2.copyMakeBorder(img, t, b, l, r, borderType=cv2.BORDER_CONSTANT, value=pad_values)
    else:
        padded = img.copy()

    return padded, (t, b, l, r)
