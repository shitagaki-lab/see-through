import numpy as np
import cv2
import torch

seg_model = None


    
def apply_segmentation(input_img, use_amp=True, s=640, model=None):
    
    if model is None:
        global seg_model
        if seg_model is None:
            from .animeseg_refine_model import load_refinenet
            seg_model = load_refinenet('animeseg')
        model = seg_model

    h0, w0 = h, w = input_img.shape[0], input_img.shape[1]
    if h > w:
        h, w = s, int(s * w / h)
    else:
        h, w = int(s * h / w), s
    ph, pw = s - h, s - w
    tmpImg = np.zeros([s, s, 3], dtype=np.float32)
    tmpImg[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h)) / 255
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = torch.from_numpy(tmpImg).unsqueeze(0).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast("cuda"):
                pred = model(tmpImg)
        else:
            pred = model(tmpImg)
        pred = pred[0][0].sigmoid().to(dtype=torch.float32)
        pred = pred[0, :, ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred.cpu().numpy().transpose((1, 2, 0)), (w0, h0))
    return pred