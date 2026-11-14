import os.path as osp
import os
from pathlib import Path
from typing import Union, List, Dict
import json
from collections.abc import MutableMapping
import gzip
from functools import lru_cache
import yaml

import numpy as np
from PIL import Image
import cv2


# numpy 2.x compatible — np.bool, np.bool8, np.float_, np.int_, np.uint were removed
NP_BOOL_TYPES = (np.bool_,)
NP_FLOAT_TYPES = (np.float16, np.float32, np.float64)
NP_INT_TYPES = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.ScalarType):
            if isinstance(obj, NP_BOOL_TYPES):
                return bool(obj)
            elif isinstance(obj, NP_FLOAT_TYPES):
                return float(obj)
            elif isinstance(obj, NP_INT_TYPES):
                return int(obj)
        return json.JSONEncoder.default(self, obj)


def json2dict(json_path: str):
    plower = json_path.lower()
    if plower.endswith('.gz'):
        with gzip.open(json_path, 'rt', encoding='utf8') as f:
            metadata = json.load(f)
        return metadata

    if plower.endswith('.yaml'):
        with open(json_path, 'r') as file:
            metadata = yaml.load(file, yaml.CSafeLoader)
        return metadata

    with open(json_path, 'r', encoding='utf8') as f:
        metadata = json.loads(f.read())
    return metadata


def serialize_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.ScalarType):
        if isinstance(obj, NP_BOOL_TYPES):
            return bool(obj)
        elif isinstance(obj, NP_FLOAT_TYPES):
            return float(obj)
        elif isinstance(obj, NP_INT_TYPES):
            return int(obj)
    return obj


def json_dump_nested_obj(obj, **kwargs):
    def _default(obj):
        if isinstance(obj, (np.ndarray, np.ScalarType)):
            return serialize_np(obj)
        return obj.__dict__
    return json.dumps(obj, default=lambda o: _default(o), ensure_ascii=False, **kwargs)


def dict2json(adict: dict, json_path: str, compress=None):
    if compress is None:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(adict, ensure_ascii=False, cls=NumpyEncoder))
    elif compress == 'gzip':
        if not json_path.endswith('.gz'):
            json_path += '.gz'
        with gzip.open(json_path, 'wt', encoding="utf8") as zipfile:
            json.dump(adict, zipfile, ensure_ascii=False, cls=NumpyEncoder)
    else:
        raise Exception(f'Invalid compression: {compress}')


IMG_EXT = ['.bmp', '.jpg', '.png', '.jpeg', '.webp', '.jxl']
def find_all_imgs(img_dir, abs_path=False, sort=False):
    imglist = []
    dir_list = os.listdir(img_dir)
    for filename in dir_list:
        if filename.startswith('.'):
            continue
        file_suffix = Path(filename).suffix
        if file_suffix.lower() not in IMG_EXT:
            continue
        if abs_path:
            imglist.append(osp.join(img_dir, filename))
        else:
            imglist.append(filename)
    if sort:
        imglist.sort()
    return imglist


def get_last_modified_file(file_prefix, exts, ext_fallback=None):
    '''
    get last modified file from files sharing same prefix
    '''
    latest_time = -1
    latest_f = None
    for ext in exts:
        tmp_p = file_prefix + ext
        if osp.exists(tmp_p) and osp.getmtime(tmp_p) > latest_time:
            latest_time = osp.getmtime(tmp_p)
            latest_f = tmp_p
    if latest_f is None:
        if ext_fallback is not None:
            latest_f = file_prefix + ext_fallback
        else:
            latest_f = file_prefix + exts[0]
    return latest_f


def find_all_files_recursive(tgt_dir: Union[List, str], ext: Union[List, set], exclude_dirs=None):
    if isinstance(tgt_dir, str):
        tgt_dir = [tgt_dir]

    if exclude_dirs is None:
        exclude_dirs = set()

    filelst = []
    for d in tgt_dir:
        for root, _, files in os.walk(d):
            if osp.basename(root) in exclude_dirs:
                continue
            for f in files:
                if Path(f).suffix.lower() in ext:
                    filelst.append(osp.join(root, f))

    return filelst


def find_all_files_with_name(tgt_dir: Union[List, str], name, exclude_dirs=None, exclude_suffix=True):
    if isinstance(tgt_dir, str):
        tgt_dir = [tgt_dir]

    if exclude_dirs is None:
        exclude_dirs = set()

    filelst = []
    for d in tgt_dir:
        for root, _, files in os.walk(d):
            if osp.basename(root) in exclude_dirs:
                continue
            for f in files:
                fn = osp.basename(f)
                if exclude_suffix:
                    fn = osp.splitext(fn)[0]
                if fn == name:
                    filelst.append(osp.join(root, f))

    return filelst


def find_all_imgs_recursive(tgt_dir, exclude_dirs=None):
    return find_all_files_recursive(tgt_dir, IMG_EXT, exclude_dirs)


VIDEO_EXT = {'.mp4', '.gif', '.webm', '.avif', '.mkv'}
def find_all_videos_recursive(tgt_dir,exclude_dirs=None):
    return find_all_files_recursive(tgt_dir, VIDEO_EXT, exclude_dirs)


def load_exec_list(exec_list, rank=None, world_size=None, check_exist=False, to_imgs=False, rank_to_worldsize=None):
    '''
    split exec_list by rank and world_size if available
    '''

    if rank_to_worldsize is not None and rank_to_worldsize != '' and rank_to_worldsize != '-':
        rank, world_size = rank_to_worldsize.split('-')
        rank = int(rank)
        world_size = int(world_size)

    if isinstance(exec_list, str):
        if osp.exists(exec_list):
            if exec_list.endswith('.json') or exec_list.endswith('.json.gz'):
                exec_list = json2dict(exec_list)
            elif exec_list.endswith('.txt'):
                with open(exec_list, 'r', encoding='utf8') as f:
                    exec_list = f.read().split('\n')
            else:
                exec_list = [exec_list]
        else:
            exec_list = exec_list.split(',')
    else:
        exec_list = list(exec_list)

    if rank is not None and world_size is not None:
        nexec = len(exec_list) // world_size
        nstart = nexec * rank
        if rank == world_size - 1:
            exec_list = exec_list[nstart:]
        else:
            exec_list = exec_list[nstart:nstart+nexec]

    if to_imgs:
        _exec_list = []
        for p in exec_list:
            if osp.isdir(p):
                _exec_list += find_all_imgs(p, sort=True, abs_path=True)
            else:
                _exec_list.append(p)
        exec_list = _exec_list

    if check_exist:
        nlist = []
        for p in exec_list:
            if osp.exists(p):
                nlist.append(p)
        return nlist
    else:
        return exec_list


def get_rank():
    if 'RANK' in os.environ:
        # print('worksize: ', os.environ['WORLD_SIZE'])
        return int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    return None, None


def load_image(imgp: str, mode="RGB", output_type='numpy'):
    """
    return RGB image as output_type
    """
    img = Image.open(imgp).convert(mode)
    if output_type == 'numpy':
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[..., None]
    return img


def flatten_dict(dictionary, parent_key='', separator='_'):
    items = []
    parent_key = str(parent_key)
    for key, value in dictionary.items():
        new_key = parent_key + separator + str(key) if parent_key else str(key)
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def imglist2imgrid(imglist, cols=4, output_type='numpy', fix_size=None):

    if isinstance(fix_size, int):
        fix_size = (fix_size, fix_size)

    current_row = []
    grid = []
    grid.append(current_row)
    for ii, img in enumerate(imglist):
        if isinstance(img, Image.Image):
            img = np.array(img)
        if fix_size is not None:
            if fix_size[0] != img.shape[0] or fix_size[1] != img.shape[1]:
                img = cv2.resize(img, (fix_size[1], fix_size[0]), interpolation=cv2.INTER_AREA)
        current_row.append(img)
        if len(current_row) >= cols and ii != len(imglist) - 1:
            current_row = []
            grid.append(current_row)
    if len(grid) > 1 and len(grid[-1]) < cols:
        for ii in range(cols - len(grid[-1])):
            grid[-1].append(np.full_like(grid[-1][-1], fill_value=255))
    if len(grid) > 1:
        for ii, row in enumerate(grid):
            grid[ii] = np.concatenate(row, axis=1)
        grid = np.concatenate(grid, axis=0)
    else:
        grid = np.concatenate(grid[0], axis=1)
    
    if output_type.lower() == 'pil':
        grid = Image.fromarray(grid)

    return grid


def pil_ensure_rgb(image: Image.Image) -> Image.Image:

    if isinstance(image, str):
        image = Image.open(image)
        
    is_array = False
    if isinstance(image, np.ndarray):
        is_array = True
        image = Image.fromarray(image)

    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

    if is_array:
        image = np.array(image)
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    padding = ((px - w) // 2, (px - h) // 2)
    canvas.paste(image, padding)
    return canvas, padding

def load_facedet_result(srcp: str):
    preds = json2dict(srcp)
    for pred in preds:
        bbox = np.array(pred['bbox'], dtype=np.float32)
        keypoints = np.array(pred['keypoints'], dtype=np.float32)
        bbox[-1] = np.round(bbox[-1] / 100)
        keypoints[:, 2] = np.round(keypoints[:, 2] / 100)
        pred['bbox'] = bbox
        pred['keypoints'] = keypoints
    return preds


def intersect_area(xyxy1, xyxy2):
    l = max(xyxy1[0], xyxy2[0])
    r = min(xyxy1[2], xyxy2[2])
    t = max(xyxy1[1], xyxy2[1])
    b = min(xyxy1[3], xyxy2[3])
    if l > r or t > b:
        return -1
    return (r - l) * (b - t)


def bbox_iou(xyxy1, xyxy2):
    i = intersect_area(xyxy1, xyxy2)
    if i < 0:
        return i
    u = (xyxy1[3] - xyxy1[1]) * (xyxy1[2] - xyxy1[0]) + \
        (xyxy2[3] - xyxy2[1]) * (xyxy2[2] - xyxy2[0]) - i
    iou = i / u
    return iou


def imread(imgpath, read_type=cv2.IMREAD_COLOR, max_retry_limit=5, retry_interval=0.1):
    if not osp.exists(imgpath):
        return None
    
    num_tries = 0
    img = Image.open(imgpath)
    if read_type != cv2.IMREAD_GRAYSCALE:
        img = img.convert('RGB')
    img = np.array(img)
    
    if read_type == cv2.IMREAD_GRAYSCALE:
        if img.ndim == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[-1] == 1:
                img = img[..., 0]
            else:
                raise
    return img


def imwrite(img_path, img, ext='.png', quality=100, jxl_encode_effort=3):
    # cv2 writing is faster than PIL
    suffix = Path(img_path).suffix
    ext = ext.lower()
    assert ext in IMG_EXT
    if suffix != '':
        img_path = img_path.replace(suffix, ext)
    else:
        img_path += ext
    encode_param = None
    if ext in {'.jpg', '.jpeg'}:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.webp':
        if quality == 100:
            quality = 101
        encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]
    if ext == '.jxl':
        # jxl_encode_effort: https://github.com/Isotr0py/pillow-jpegxl-plugin/issues/23
        # higher values theoretically produce smaller files at the expense of time, 3 seems to strike a balance
        lossless = quality > 99 # quality=100, lossless=False seems to result in larger file compared with lossless=True
        Image.fromarray(img).save(img_path, quality=quality, lossless=lossless, effort=jxl_encode_effort)
    else:
        if len(img.shape) == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imencode(ext, img, encode_param)[1].tofile(img_path)


def save_tmp_img(img: Union[Image.Image, np.ndarray], savep = 'local_tst.png', mask2img=False):
    '''
    for debug img output
    '''
    if not savep:
        savep = 'local_tst.png'
    if mask2img:
        img = img.astype(np.uint8) * 255
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        img = Image.fromarray(img.astype(np.uint8))
    img.save(savep)


def bbox2xyxy(box):
    x1, y1 = box[0], box[1]
    return x1, y1, x1+box[2], y1+box[3]


def bbox_overlap_area(abox, boxb) -> int:
    ax1, ay1, ax2, ay2 = bbox2xyxy(abox)
    bx1, by1, bx2, by2 = bbox2xyxy(boxb)
    
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    
    if ix > 0 and iy > 0:
        return ix * iy
    else:
        return 0


def bbox_overlap_xy(abox, boxb):
    ax1, ay1, ax2, ay2 = bbox2xyxy(abox)
    bx1, by1, bx2, by2 = bbox2xyxy(boxb)
    
    ix = min(ax2, bx2) - max(ax1, bx1)
    iy = min(ay2, by2) - max(ay1, by1)
    
    return ix, iy


@lru_cache(maxsize=4)
def get_all_segcls(cls_path: str):
    if cls_path.lower().endswith('.json'):
        cls_list = list(json2dict(cls_path).keys())
    else:
        with open(cls_path, 'r', encoding='utf8') as f:
            c = f.read()
            cls_list = [l.strip() for l in c.split('\n') if l.strip()]
    return cls_list


def imglist_from_dir_or_flist(src):
    if osp.isdir(src):
        lst = find_all_imgs(src, sort=True, abs_path=True)
    else:
        assert osp.isfile(src)
        lst = load_exec_list(src)
    return lst


def find_closest_point_from_line2(p0, p1, pts):
    dist = np.linalg.norm(pts - p0[None, :], axis=1) + np.linalg.norm(pts - p1[None, :], axis=1)
    return pts[np.argsort(dist)[0]]


def cosine_similarity_numpy(v1, v2):
    """Calculates the cosine similarity between two NumPy vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0  # Or handle as an error, depending on requirements
    
    return dot_product / (magnitude_v1 * magnitude_v2)


def xyxy2center(xyxy):
    xyxy = np.array(xyxy)
    return np.array(xyxy[[0, 1]] + xyxy[[2, 3]]) / 2


def save_psd(savep, img_list, h, w, pad_to_canvas=False, mode='RGBA', img_key='img'):
    from psd_tools import PSDImage
    psd_image = PSDImage.new(mode=mode, size=(w, h), depth=8)
    for imgd in img_list:
        img = imgd[img_key]
        x1 = y1 = 0
        if 'xyxy' in imgd:
            x1, y1, x2, y2 = imgd['xyxy']
        if 'xyxy' in imgd and pad_to_canvas:
            img_padded = np.zeros((h, w, 4), dtype=np.uint8)
            img_padded[y1: y2, x1: x2] = img
            x1 = y1 = 0
            img = img_padded
        img = Image.fromarray(img)
        layer_name = 'undefined'
        if 'layer_name' in imgd:
            layer_name = imgd['layer_name']
        elif 'tag' in imgd:
            layer_name = imgd['tag']
        psd_image.create_pixel_layer(img, name=layer_name, top=y1, left=x1, opacity=255)

    psd_image.save(savep)

def load_part(srcp: str, rotate=False, pad=0, min_width=64, min_sz=12, depth_min=None, depth_max=None):
    img = Image.open(srcp).convert('RGBA')
    srcd = osp.dirname(srcp)
    tag = osp.splitext(osp.basename(srcp))[0]
    depthp = osp.join(srcd, tag + '_depth.png')
    tag_infop = osp.join(srcd, tag + '.json')
    
    img = np.array(img)
    p_test = max(img.shape[:2]) // 10
    mask = img[...,  -1] > 10

    if isinstance(pad, int):
        pad = [pad] * 4

    if osp.exists(tag_infop):
        rst = json2dict(tag_infop)
        depth = np.array(Image.open(depthp).convert('L'))
        depth = np.array(depth, dtype=np.float32) / 255
        rst.update({'img': img, 'depth': depth, 'mask': mask, 'tag': tag})
        return rst

    if np.sum(mask[:-p_test, :-p_test]) > 4:
        if rotate:
            img = np.rot90(img, 3)
            mask = np.rot90(mask, 3, )

        xyxy = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
        xyxy = np.array(xyxy)
        h, w = xyxy[2:]
        xyxy[2] += xyxy[0]
        xyxy[3] += xyxy[1]
        p = min_width - w
        if p > 0:
            if xyxy[0] >= p:
                xyxy[0] -= p
            else:
                xyxy[2] += p
        p = min_sz - h
        if p > 0:
            if xyxy[1] >= p:
                xyxy[1] -= p
            else:
                xyxy[3] += p

        depth = np.array(Image.open(depthp).convert('L'))
        if rotate:
            depth = np.rot90(depth, 3)

        x1, y1, x2, y2 = xyxy
        mask = mask[y1: y2, x1: x2].copy()
        img = img[y1: y2, x1: x2].copy()
        depth = depth[y1: y2, x1: x2].copy()

        pt, pb, pl, pr = pad
        if pt > 0 or pb > 0 or pl > 0 or pr > 0:
            img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            depth = cv2.copyMakeBorder(depth, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(255))
            mask = cv2.copyMakeBorder(mask.astype(np.uint8), pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0)) > 0
            x1 -= pl
            y1 -= pt
            x2 += pr
            y2 += pb
            xyxy = [x1, y1, x2, y2]
        
        # dmin, dmax = partdict['depth_min'], partdict['depth_max']
        depth = np.array(depth, dtype=np.float32) / 255
        rst =  {'img': img, 'depth': depth, 'xyxy': xyxy, 'mask': mask, 'tag': tag}
        if depth_max is not None and depth_min is not None:
            dmax, dmin = depth_max, depth_min
            depth = depth * (dmax - dmin) + dmin
            rst.update({'depth': depth, 'depth_min': dmin, 'depth_max': dmax})

        return rst
    else:
        return None

def load_img_depth(srcd, src_info, pad=5, try_crop=False, rotate=False):
    '''
    pad: int or [pt, pb, pl, pr]
    '''
    tag2infos = src_info['parts']

    if isinstance(pad, int):
        pad = [pad] * 4

    for t in tag2infos:
        part_info = tag2infos[t]
        if 'img' in part_info:
            continue
        img = np.array(Image.open(osp.join(srcd, t + '.png')).convert('RGBA'))
        # if rotate:
        #     img = np.rot90
        # if try_crop:
        #     xyxy = cv2.boundingRect(cv2.findNonZero((img[..., -1] > 10).astype(np.uint8)))
        depth = np.array(Image.open(osp.join(srcd, t + '_depth.png')).convert('L'))
        if 'depth_max' in part_info:
            dmax, dmin = part_info['depth_max'], part_info['depth_min']
            depth = np.array(depth, dtype=np.float32) / 255 * (dmax - dmin) + dmin
        else:
            depth = np.array(depth, dtype=np.float32) / 255

        if 'xyxy' in part_info:
            x1, y1, x2, y2 = part_info['xyxy']
            pt, pb, pl, pr = pad
            if pt > 0 or pb > 0 or pl > 0 or pr > 0:
                img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
                depth = cv2.copyMakeBorder(depth, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(1))
                x1 -= pl
                y1 -= pt
                x2 += pr
                y2 += pb
                part_info['xyxy'] = [x1, y1, x2, y2]

        part_info['depth'] = depth
        part_info['img'] = img
        part_info['mask'] = (img[..., -1] > 10).astype(np.uint8) * 255


def load_parts(srcp, rotate=False, pad=0, min_width=64):
    srcimg = osp.join(srcp, 'src_img.png')
    fullpage = np.array(Image.open(srcimg).convert('RGBA'))

    infop = osp.join(srcp, 'info.json')
    infos = json2dict(infop)

    part_dict_list = []
    tag2pd = {}
    part_id = 0

    min_sz = 12
    if isinstance(pad, int):
        pad = [pad] * 4

    if rotate:
        fullpage = np.rot90(fullpage, 3, )

    for tag, partdict in infos['parts'].items():

        # img = Image.open(osp.join(srcp, tag + '.png')).convert('RGBA')
        # depthp = osp.join(srcp, tag + '_depth.png')
        
        # img = np.array(img)
        # p_test = max(img.shape[:2]) // 10
        # mask = img[...,  -1] > 10
        # if np.sum(mask[:-p_test, :-p_test]) > 4:
        #     if rotate:
        #         img = np.rot90(img, 3)
        #         mask = np.rot90(mask, 3, )

        #     xyxy = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
        #     xyxy = np.array(xyxy)
        #     h, w = xyxy[2:]
        #     xyxy[2] += xyxy[0]
        #     xyxy[3] += xyxy[1]
        #     p = min_width - w
        #     if p > 0:
        #         if xyxy[0] >= p:
        #             xyxy[0] -= p
        #         else:
        #             xyxy[2] += p
        #     p = min_sz - h
        #     if p > 0:
        #         if xyxy[1] >= p:
        #             xyxy[1] -= p
        #         else:
        #             xyxy[3] += p

        #     depth = np.array(Image.open(depthp).convert('L'))
        #     if rotate:
        #         depth = np.rot90(depth, 3)

        #     x1, y1, x2, y2 = xyxy
        #     mask = mask[y1: y2, x1: x2].copy()
        #     img = img[y1: y2, x1: x2].copy()
        #     depth = depth[y1: y2, x1: x2].copy()

        #     pt, pb, pl, pr = pad
        #     if pt > 0 or pb > 0 or pl > 0 or pr > 0:
        #         img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        #         depth = cv2.copyMakeBorder(depth, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(255))
        #         mask = cv2.copyMakeBorder(mask.astype(np.uint8), pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0)) > 0
        #         x1 -= pl
        #         y1 -= pt
        #         x2 += pr
        #         y2 += pb
        #         xyxy = [x1, y1, x2, y2]
            
        #     dmin, dmax = partdict['depth_min'], partdict['depth_max']
        #     depth = np.array(depth, dtype=np.float32) / 255 * (dmax - dmin) + dmin
            p = load_part(osp.join(srcp, tag + '.png'), rotate=rotate, pad=pad, min_width=min_width, min_sz=min_sz)
            if p is not None:
                tag2pd[tag] = p
                tag2pd[tag]['part_id'] = part_id
                part_dict_list.append(tag2pd[tag])
                part_id += 1
            

    return fullpage, infos, part_dict_list

