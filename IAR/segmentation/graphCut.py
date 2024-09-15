import cv2
import numpy as np



def quantizeSegment(mask_img):

    mask_img[mask_img<=64] = 0
    mask_img[(mask_img>64)&(mask_img<=192)] = 128
    mask_img[mask_img>192] = 255

    return mask_img

def graphCut(input_img, mask_img, n_iter=15):

    # - Quantize mask image
    mask_img = quantizeSegment(mask_img)    

    # - Set flags to the mask
    mask = np.zeros_like(mask_img, np.uint8)
    mask[mask_img==0] = cv2.GC_BGD # definite background
    mask[mask_img==128] = cv2.GC_PR_BGD # possible background
    mask[mask_img==255] = cv2.GC_FGD # definite foreground

    # - Allocate memory for the background and foreground models
    bgd_model = np.zeros((1,65), np.float64)
    fgd_model = np.zeros((1,65), np.float64)

    # - Apply grabCut
    seg_img, _, _ = cv2.grabCut(input_img, mask, None, bgd_model, fgd_model,
    n_iter, cv2.GC_INIT_WITH_MASK)

    # - Combine flags
    seg_img = np.where((seg_img==cv2.GC_FGD)|(seg_img==cv2.GC_PR_FGD),1,0).astype(np.uint8)

    return seg_img