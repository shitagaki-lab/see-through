import os
import os.path as osp

from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline, UNetFrameConditionModel
from modules.layerdiffuse.vae import TransparentVAE
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.marigold import MarigoldDepthPipeline
from utils.cv import center_square_pad_resize, img_alpha_blending, smart_resize, validate_resolution
from utils.torch_utils import seed_everything
from utils.io_utils import json2dict, dict2json, load_parts, save_tmp_img, load_part, save_psd
from utils.torchcv import cluster_inpaint_part

from psd_tools import PSDImage
from safetensors.torch import load_file
import cv2
import numpy as np
import torch
from PIL import Image

VALID_BODY_PARTS_V2 = [
    'hair', 'headwear', 'face', 'eyes', 'eyewear', 'ears', 'earwear', 'nose', 'mouth', 
    'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 
    'tail', 'wings', 'objects'
]


layerdiff_pipeline: KDiffusionStableDiffusionXLPipeline = None
def apply_layerdiff(
    imgp: str, pretrained: str, num_inference_steps=30, seed=0, save_dir='workspace/layerdiff_output', target_tag_list=VALID_BODY_PARTS_V2, 
    resolution=1280, vae_ckpt=None, unet_ckpt=None, disable_progressbar=False):
    
    global layerdiff_pipeline
    if layerdiff_pipeline is None:
        trans_vae = TransparentVAE.from_pretrained(pretrained, subfolder='trans_vae')
        if unet_ckpt is None:
            unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet')
        else:
            print(f'load unet from {unet_ckpt}')
            unet = UNetFrameConditionModel.from_pretrained(unet_ckpt)
        layerdiff_pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
            pretrained,
            trans_vae=trans_vae, unet=unet,
            scheduler=None
        )

        if vae_ckpt is not None:
            td_sd = {}
            vae_sd = {}
            sd = load_file(vae_ckpt)
            for k, v in sd.items():
                if k.startswith('trans_decoder.'):
                    td_sd[k.lstrip('trans_decoder.')] = v
                elif k.startswith('vae.'):
                    vae_sd[k.replace('vae.', '')] = v

            if len(vae_sd) > 0:
                layerdiff_pipeline.vae.load_state_dict(vae_sd)
                print(f'load vae from {vae_ckpt}')

            if len(td_sd) > 0:
                layerdiff_pipeline.trans_vae.decoder.load_state_dict(td_sd)
                print(f'load vae from {vae_ckpt}')

        layerdiff_pipeline.vae.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.trans_vae.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.unet.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.text_encoder.to(dtype=torch.bfloat16, device='cuda')
        layerdiff_pipeline.text_encoder_2.to(dtype=torch.bfloat16, device='cuda')

    pipeline = layerdiff_pipeline
    pipeline.set_progress_bar_config(disable=disable_progressbar)

    saved = osp.join(save_dir, osp.splitext(osp.basename(imgp))[0])
    os.makedirs(saved, exist_ok=True)
    input_img = np.array(Image.open(imgp).convert('RGBA'))
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, resolution, return_pad_info=True)
    scale = pad_size[0] / resolution
    Image.fromarray(fullpage).save(osp.join(saved, 'src_img.png'))

    rng = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    tag_version = pipeline.unet.get_tag_version()
    if tag_version == 'v2':
        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=target_tag_list,
            negative_prompt='',
            fullpage=fullpage
        )
        images = pipeline_output.images
        for rst, tag in zip(images, target_tag_list):
            savename = osp.join(saved, f'{tag}.png')
            Image.fromarray(rst).save(savename)

    elif tag_version == 'v3':

        def _crop_head(img, xywh):
            x, y, w, h = xywh
            ih, iw = img.shape[:2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            if w < iw // 2:
                px = min(iw - x - w, x, w // 5)
                x1 = min(max(x - px, 0), iw)
                x2 = min(max(x + w + px, 0), iw)
            if h < ih // 2:
                py = min(ih - y - h, y, h // 5)
                y2 = min(max(y + h + py, 0), ih)
                y1 = min(max(y - py, 0), ih)

            return img[y1: y2, x1: x2], (x1, y1, x2, y2)

        body_tag_list = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=body_tag_list,
            negative_prompt='',
            fullpage=fullpage,
            group_index=0
        )
        images = pipeline_output.images
        for rst, tag in zip(pipeline_output.images, body_tag_list):
            savename = osp.join(saved, f'{tag}.png')
            Image.fromarray(rst).save(savename)
        head_img = images[2]
        # head_img = np.array(Image.open(osp.join(saved, 'head.png')))

        head_tag_list = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
        hx0, hy0, hw, hh = cv2.boundingRect(cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8)))
        
        hx = int(hx0 * scale) - pad_pos[0]
        hy = int(hy0 * scale) - pad_pos[1]
        hw = int(hw * scale)
        hh = int(hh * scale)
        input_head, (hx1, hy1, hx2, hy2) = _crop_head(input_img, [hx, hy, hw, hh])
        hx1 = int(hx1 / scale + pad_pos[0] / scale)
        hy1 = int(hy1 / scale + pad_pos[1] / scale)
        ih, iw = input_head.shape[:2]
        input_head, pad_size, pad_pos = center_square_pad_resize(input_head, resolution, return_pad_info=True)

        Image.fromarray(input_head).save(osp.join(saved, 'src_head.png'))

        pipeline_output = pipeline(
            strength=1.0,
            num_inference_steps=num_inference_steps,
            batch_size=1,
            generator=rng,
            guidance_scale=1.0,
            prompt=head_tag_list,
            negative_prompt='',
            fullpage=input_head,
            group_index=1
        )
        canvas = np.zeros((resolution, resolution, 4), dtype=np.uint8)

        py1, py2, px1, px2 = (np.array([pad_pos[1], pad_pos[1] + ih, pad_pos[0], pad_pos[0] + iw]) / scale).astype(np.int64)
        
        scale_size = (int(pad_size[0] / scale), int(pad_size[1] / scale))

        for rst, tag in zip(pipeline_output.images, head_tag_list):
            rst = smart_resize(rst, scale_size)[py1: py2, px1: px2]
            full = canvas.copy()
            full[hy1: hy1 + rst.shape[0], hx1: hx1 + rst.shape[1]] = rst
            savename = osp.join(saved, f'{tag}.png')
            Image.fromarray(full).save(savename)

    else:
        raise




marigold_pipeline: MarigoldDepthPipeline = None
def apply_marigold(srcp, pretrained: str, num_inference_steps=-1, seed=0, save_dir='workspace/layerdiff_output', target_tag_list=VALID_BODY_PARTS_V2, resolution=1280, normalize_depth=False, disable_progressbar=False):
    global marigold_pipeline
    if marigold_pipeline is None:
        unet = UNetFrameConditionModel.from_pretrained(pretrained, subfolder='unet')
        marigold_pipeline = MarigoldDepthPipeline.from_pretrained(pretrained, unet=unet)
        marigold_pipeline.to(device='cuda', dtype=torch.bfloat16)
    pipe = marigold_pipeline
    pipe.set_progress_bar_config(disable=disable_progressbar)

    srcname = osp.basename(osp.splitext(srcp)[0])
    saved = osp.join(save_dir, srcname)

    src_img_p = osp.join(saved, 'src_img.png')
    fullpage = np.array(Image.open(src_img_p).convert('RGBA'))

    src_h, src_w = fullpage.shape[:2]
    if isinstance(resolution, int) and resolution == -1:
        resolution = [src_h, src_w]
    resolution = validate_resolution(resolution)
    src_rescaled = resolution[0] != src_h or resolution[1] != src_w

    img_list = []
    caption_list = []
    exist_list = []
    empty_array = np.zeros((src_h, src_w, 4), dtype=np.uint8)
    blended_alpha = np.zeros((src_h, src_w), dtype=np.float32)

    compose_list = {'eyes': ['eyewhite', 'irides', 'eyelash', 'eyebrow'], 'hair': ['back hair', 'front hair']}
    for tag in VALID_BODY_PARTS_V2:
        tagp = osp.join(saved, f'{tag}.png')
        if osp.exists(tagp):
            exist_list.append(True)
            caption_list.append(tag)
            tag_arr = np.array(Image.open(tagp))
            tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
            # blended_alpha += tag_arr[..., -1].astype(np.float32) / 255
            img_list.append(tag_arr)
        else:
            img_list.append(empty_array)
            exist_list.append(False)

    compose_dict = {}
    for c, clist in compose_list.items():
        imlist = []
        taglist = []
        for tag in clist:
            p = osp.join(saved, tag + '.png')
            if osp.exists(p):
                tag_arr = np.array(Image.open(p))
                tag_arr[..., -1][tag_arr[..., -1] < 15] = 0
                imlist.append(tag_arr)
                taglist.append(tag)
        if len(imlist) > 0:
            img = img_alpha_blending(imlist, premultiplied=False)
            img_list[VALID_BODY_PARTS_V2.index(c)] = img
            compose_dict[c] = {'taglist': taglist, 'imlist': imlist}

    for img in img_list:
        blended_alpha += img[..., -1].astype(np.float32) / 255

    blended_alpha = np.clip(blended_alpha, 0, 1) * 255
    blended_alpha = blended_alpha.astype(np.uint8)
    fullpage[..., -1] = blended_alpha
    img_list.append(fullpage)

    img_list_input = img_list
    if src_rescaled:
        img_list_input = [smart_resize(img, resolution) for img in img_list]

    seed_everything(seed)
    pipe_out = pipe(
        # tensor2img(img, 'pil', denormalize=True, mean=127.5, std=127.5),
        color_map=None,
        img_list = img_list_input,
        denoising_steps=num_inference_steps
    )
    depth_pred: np.ndarray = pipe_out.depth_tensor
    
    depth_pred = depth_pred.to(device='cpu', dtype=torch.float32).numpy()
    if src_rescaled:
        depth_pred = [smart_resize(d, (src_h, src_w)) for d in depth_pred]
    drawables = [{'img': img, 'depth': depth} for img, depth in zip(img_list, depth_pred)]
    drawables = drawables[:-1]
    blended = img_alpha_blending(drawables, premultiplied=False)

    infop = osp.join(saved, 'info.json')
    if osp.exists(infop):
        info = json2dict(infop)
    else:
        info = {'parts': {}}

    parts = info['parts']
    for ii, depth in enumerate(depth_pred[:-1]):
        if normalize_depth:
            depth_max, depth_min = depth.max(), depth.min()
            depth = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-7) * 255, 0, 255).astype(np.uint8)
        else:
            depth = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        # depth = depth[..., None][..., [-1] * 3].copy()
        tag = VALID_BODY_PARTS_V2[ii]
        if tag in compose_dict:
            mask = blended_alpha > 256
            for t, im in zip(compose_dict[tag]['taglist'][::-1], compose_dict[tag]['imlist'][::-1]):
                mask_local = im[..., -1] > 15
                mask_invis = np.bitwise_and(mask, mask_local)
                depth_local = np.full((src_h, src_w), fill_value=255, dtype=np.uint8)
                depth_local[mask_local] = depth[mask_local]
                if np.any(mask_invis):
                    depth_local[mask_invis] = np.median(depth[np.bitwise_and(mask_local, np.bitwise_not(mask_invis))])
                mask = np.bitwise_or(mask, mask_local)

                parts_info = parts.get(t, {})
                savep = osp.join(saved, f'{t}_depth.png')
                Image.fromarray(depth_local).save(savep)
                parts[t] = parts_info
                if normalize_depth:
                    parts_info['depth_max'] = depth_max
                    parts_info['depth_min'] = depth_min
            continue

        parts_info = parts.get(tag, {})
        savep = osp.join(saved, f'{tag}_depth.png')
        Image.fromarray(depth).save(savep)
        parts[tag] = parts_info
        if normalize_depth:
            parts_info['depth_max'] = depth_max
            parts_info['depth_min'] = depth_min

    dict2json(info, infop)
    Image.fromarray(blended).save(osp.join(saved, 'reconstruction.png'))


def label_lr_split(labels, stats, id1, id2):
    label1 = (labels == id1).astype(np.uint8) * 255
    label2 = (labels == id2).astype(np.uint8) * 255

    stats1, stats2 = stats[id1], stats[id2]

    x1 = stats[id1][0] + stats[id1][2] / 2
    x2 = stats[id2][0] + stats[id2][2] / 2

    if x2 < x1:
        return label2, label1, stats2, stats1
    else:
        return label1, label2, stats1, stats2


def save_part(tag, saved, part_dict, crop=True, save_part_info=False, save_to_disk=True):
    img = part_dict.pop('img')
    if 'mask' in part_dict:
        part_dict.pop('mask')

    depth = part_dict.pop('depth')
    mask = img[..., -1] > 10
    depth_median = np.median(depth[mask])

    if crop:
        xywh = cv2.boundingRect(cv2.findNonZero(mask.astype(np.uint8)))
        xyxy = np.array(xywh).copy()
        xyxy[2] += xyxy[0]
        xyxy[3] += xyxy[1]
        depth = depth[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]
        img = img[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]

        x1, y1, x2, y2 = part_dict['xyxy']
        part_dict['xyxy'] = [x1 + xyxy[0], y1 + xyxy[1], x1 + xyxy[2], y1 + xyxy[3]]

    # dmin, dmax = np.min(depth), np.max(depth)
    depth = np.clip(depth, 0, 1) * 255
    depth = np.round(depth).astype(np.uint8)


    # part_dict['depth_min'] = dmin
    # part_dict['depth_max'] = dmax
    part_dict['depth_median'] = depth_median
    part_dict['tag'] = tag

    if save_to_disk:
        Image.fromarray(img).save(osp.join(saved, tag + '.png'))
        Image.fromarray(depth).save(osp.join(saved, tag + '_depth.png'))
        if save_part_info:
            dict2json(part_dict, osp.join(saved, tag + '.json'))
    else:
        part_dict['img'] = img
        part_dict['depth'] = depth 

    return part_dict


def process_cuts(img, depth, src_xyxy, tgt_bbox, p=5, mask=None):
    tx1, ty1, tx2, ty2 = tgt_bbox[:4]
    tx2 += tx1
    ty2 += ty1
    img = img[ty1: ty2, tx1: tx2].copy()
    depth = depth[ty1: ty2, tx1: tx2]
    
    depth_median = 1
    if mask is not None:
        mask = (mask[ty1: ty2, tx1: tx2].copy() > 15).astype(np.uint8)
        ksize = 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
        mask = cv2.dilate(mask, element)
        img[..., -1] *= mask
        depth = 1 - (1-depth) * mask
        if np.any(mask):
            depth_median = np.median(depth[mask])
    
    fxyxy = [tx1 + src_xyxy[0], ty1 + src_xyxy[1], tx2 + src_xyxy[0], ty2 + src_xyxy[1]]

    return img, depth, fxyxy, depth_median


def part_lr_split(tag, part_info):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        part_info['mask'].astype(np.uint8) * 255, connectivity=8)
    tag2pinfo = {}
    if len(stats) > 2:
        stats = np.array(stats)
        stats_order = np.argsort(stats[..., -1])[::-1][1:]
        arml_mask, armr_mask, statsl, statsr = label_lr_split(labels, stats, stats_order[0], stats_order[1])
        depth_median = part_info.get('depth_median', 1)

        img, depth, xyxy, depth_median = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsl, mask=arml_mask)
        arml_mask = arml_mask[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]
        tag2pinfo[f'{tag}-r'] = {'img': img, 'xyxy': xyxy, 'depth': depth, 'depth_median': depth_median, 'tag': f'{tag}-r'}

        img, depth, xyxy, depth_median = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsr, mask=armr_mask)
        armr_mask = armr_mask[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]]
        tag2pinfo[f'{tag}-l'] = {'img': img, 'xyxy': xyxy, 'depth': depth, 'depth_median': depth_median, 'tag': f'{tag}-l'}

    else:
        tag2pinfo[tag] = part_info
    return tag2pinfo


def tag_lr_split(tag: str, tag2pinfo):
    if tag in tag2pinfo:
        part_info = tag2pinfo.pop(tag)
        tag2pinfo.update(part_lr_split(tag, part_info))


def further_extr(srcd: str, rotate=True, save_to_psd=False, tblr_split=True):


    saved = osp.join(srcd, 'optimized')
    # infos = json2dict(osp.join(srcd, 'info.json'))
    os.makedirs(saved, exist_ok=True)

    fullpage, infos, part_dict_list = load_parts(srcd, rotate=rotate)

    # optim_depth(part_dict_list, fullpage)

    tag2pinfo = {}
    for pinfo in part_dict_list:
        tag = pinfo['tag']
        tag2pinfo[tag] = pinfo

    if 'eyes' in tag2pinfo:
        part_info = tag2pinfo.pop('eyes')
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            part_info['mask'].astype(np.uint8) * 255, connectivity=8)
        if len(stats) > 2:
            stats = np.array(stats)
            
            if len(stats[..., -1]) >= 5:
                stats_order = np.argsort(stats[..., -1])[::-1][1:]
                eyel_mask, eyer_mask, statsl, statsr = label_lr_split(labels, stats, stats_order[0], stats_order[1])
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsl)
                tag2pinfo['eyer'] = {'img': img, 'xyxy': xyxy, 'depth': depth}
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsr)
                tag2pinfo['eyel'] = {'img': img, 'xyxy': xyxy, 'depth': depth}

                browl_mask, browr_mask, statsl, statsr = label_lr_split(labels, stats, stats_order[2], stats_order[3])
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsl)
                tag2pinfo['browr'] = {'img': img, 'xyxy': xyxy, 'depth': depth}
                img, depth, xyxy, _ = process_cuts(part_info['img'], part_info['depth'], part_info['xyxy'], statsr)
                tag2pinfo['browl'] = {'img': img, 'xyxy': xyxy, 'depth': depth}
        else:
            tag2pinfo['eyes'] = part_info


    if tblr_split:
        tag_lr_split('handwear', tag2pinfo)

        eyetags_v3 = ['eyewhite', 'irides', 'eyelash', 'eyebrow']
        for tag in eyetags_v3:
            tag_lr_split(tag, tag2pinfo)

        tag_lr_split('ears', tag2pinfo)

    # if 'headwear' in tag2pinfo:
    #     part_info = tag2pinfo.pop('headwear')
    #     tag2pinfo['hair']['img'] = img_alpha_blending([tag2pinfo['hair'], part_info], xyxy=tag2pinfo['hair']['xyxy'], premultiplied=False)
    # if 'headwear' in tag2pinfo:
    #     part_info = tag2pinfo.pop('headwear')
    #     tag2pinfo['hair']['img'] = img_alpha_blending([tag2pinfo['hair'], part_info], xyxy=tag2pinfo['hair']['xyxy'], premultiplied=False)

    # if 'footwear' in tag2pinfo:
    #     part_info = tag2pinfo.pop('footwear')
    #     tag2pinfo['legwear']['img'] = img_alpha_blending([tag2pinfo['legwear'], part_info], xyxy=tag2pinfo['legwear']['xyxy'], premultiplied=False)

        if 'hair' in tag2pinfo:
            part_info = tag2pinfo.pop('hair')
            parts = cluster_inpaint_part(**part_info)
            parts.sort(key=lambda x: x['depth_median'])
            tag2pinfo['hairf'] = parts[0]
            tag2pinfo['hairb'] = parts[1]

    # if 'footwear' in tag2pinfo:
    #     tag2pinfo.pop('footwear')

    if 'nose' in tag2pinfo:
        xyxy = tag2pinfo['nose']['xyxy']
        tag2pinfo['nose']['img'][..., :3] = fullpage[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2], :3]

    if 'mouth' in tag2pinfo:
        xyxy = tag2pinfo['mouth']['xyxy']
        tag2pinfo['mouth']['img'][..., :3] = fullpage[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2], :3]

    part_dict_list = []
    save_dir = osp.dirname(saved)
    psd_savep = osp.join(osp.dirname(save_dir), osp.basename(save_dir) + '.psd')

    for t in tag2pinfo:
        if t not in tag2pinfo:
            print(f'{t} is not valid')
            continue
        part_dict = tag2pinfo[t]
        part_dict = save_part(t, saved, part_dict, save_to_disk=not save_to_psd)
        if save_to_psd:
            part_dict_list.append(part_dict)

    if 'face' in tag2pinfo:
        for t in ['nose', 'mouth', 'eyes']:
            if t in tag2pinfo:
                if tag2pinfo[t]['depth_median'] > tag2pinfo['face']['depth_median']:
                    tag2pinfo[t]['depth_median'] = tag2pinfo['face']['depth_median'] - 0.001
        for t in ['earr', 'earl', 'ears']:
            if t in tag2pinfo:
                tag2pinfo[t]['depth_median'] = tag2pinfo['face']['depth_median'] + 0.001

    # if 'hairb' in tag2pinfo:
    #     tag2pinfo['hairb']['depth_median'] = 1.

    frame_size = fullpage.shape[:2]

    if save_to_psd:
        dump_parts_psd(tag2pinfo, frame_size, psd_savep, part_dict_list=part_dict_list)
        print(f'psd saved to {psd_savep}')
    else:
        dict2json({'parts': tag2pinfo, 'frame_size': frame_size}, osp.join(saved, 'info.json'))


def dump_parts_psd(tag2pinfo, frame_size, psd_savep, part_dict_list=None):
    if part_dict_list is None:
        part_dict_list = []
        for v in tag2pinfo.values():
            part_dict_list.append(v)
    psd_depth_savep = osp.splitext(psd_savep)[0] + '_depth.psd'
    part_dict_list.sort(key=lambda x: x['depth_median'], reverse=True)
    save_psd(psd_savep, part_dict_list, frame_size[0], frame_size[1])
    save_psd(psd_depth_savep, part_dict_list, frame_size[0], frame_size[1], mode='L', img_key='depth')
    for pdict in tag2pinfo.values():
        for k in {'img', 'depth', 'mask'}:
            if k in pdict:
                pdict.pop(k)
    dict2json({'parts': tag2pinfo, 'frame_size': frame_size}, psd_savep + '.json')


def psd2partdicts(srcp):
    psd = PSDImage.open(srcp)
    json_path = srcp + '.json'
    partdict = json2dict(json_path)
    tag2part= partdict['parts']
    for layer in psd:
        img = layer.numpy()
        tag2part[layer.name]['img'] = np.round(img * 255).astype(np.uint8)
    depth_path = osp.splitext(srcp)[0] + '_depth.psd'
    psd = PSDImage.open(depth_path)
    for layer in psd:
        img = layer.numpy()
        tag2part[layer.name]['depth'] = img[..., 0]
    return partdict


def seg_wdepth(srcp, *args, **kwargs):
    srcd = osp.dirname(srcp)
    part_dict = load_part(srcp)
    tag = part_dict['tag']
    rst_list = cluster_inpaint_part(**part_dict)
    saved = osp.join(srcd, tag)
    if len(rst_list) > 0:
        os.makedirs(saved, exist_ok=True)
        for ii, part in enumerate(rst_list):
            sub_tag = tag + '-' + str(ii)
            save_part(sub_tag, saved, part, save_part_info=True)
        print(f'sub part saved to {saved}')
    else:
        print(f'seg_wdepth: failed to seg more parts')


def seg_wdepth_psd(srcp, target_tags, savep=None):
    part_infos = psd2partdicts(srcp)
    if savep is None:
        savep = osp.splitext(srcp)[0] + '_wdepth.psd'
    if isinstance(target_tags, str):
        target_tags = target_tags.split(',')
    else:
        assert isinstance(target_tags, list)
    valid_tags = list(part_infos['parts'].keys())
    for tag in target_tags:
        if tag not in part_infos['parts']:
            print(f'{tag} is not in {valid_tags}')
            continue
        part_dict = part_infos['parts'].pop(tag)
        mask = part_dict['img'][..., -1] > 10
        if not np.any(mask):
            continue
        part_dict['mask'] = mask
        rst_list = cluster_inpaint_part(**part_dict)
        if len(rst_list) > 0:
            for ii, part in enumerate(rst_list):
                sub_tag = tag + '-' + str(ii)
                part['tag'] = sub_tag
                part_infos['parts'][sub_tag] = part

    dump_parts_psd(part_infos['parts'], part_infos['frame_size'], savep)
    print(f'psd saved to {savep}')


def seg_wlr(srcp, *args, **kwargs):
    srcd = osp.dirname(srcp)
    part_dict = load_part(srcp)
    tag = part_dict['tag']
    rst_dict = part_lr_split(tag, part_dict)
    saved = osp.join(srcd, tag)
    if len(rst_dict) > 1:
        os.makedirs(saved, exist_ok=True)
        for sub_tag, part in rst_dict.items():
            save_part(sub_tag, saved, part, save_part_info=True)
        print(f'sub part saved to {saved}')
    else:
        print(f'seg_wdepth: failed to seg more parts')



def seg_wlr_psd(srcp, target_tags, savep=None):
    part_infos = psd2partdicts(srcp)
    if savep is None:
        savep = osp.splitext(srcp)[0] + '_lrsplit.psd'
    if isinstance(target_tags, str):
        target_tags = target_tags.split(',')
    else:
        assert isinstance(target_tags, list)
    valid_tags = list(part_infos['parts'].keys())
    for tag in target_tags:
        if tag not in part_infos['parts']:
            print(f'{tag} is not in {valid_tags}')
            continue
        part_dict = part_infos['parts'].pop(tag)
        mask = part_dict['img'][..., -1] > 10
        if not np.any(mask):
            continue
        part_dict['mask'] = mask
        rst_dict = part_lr_split(tag, part_dict)
        part_infos['parts'].update(rst_dict)

    dump_parts_psd(part_infos['parts'], part_infos['frame_size'], savep)
    print(f'psd saved to {savep}')