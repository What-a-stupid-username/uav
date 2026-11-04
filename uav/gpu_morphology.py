"""Exact binary dilation footprints used by existing CPU mask processing."""
import torch
import torch.nn.functional as F
from .mask_geometry import FINAL_MASK_RADIUS, FINAL_MASK_KERNEL_SIZE


def dilate_cross(mask, iterations):
    result=mask.bool()
    for _ in range(iterations):
        padded=F.pad(result,(1,1,1,1))
        result=(padded[...,1:-1,1:-1]|padded[...,:-2,1:-1]|padded[...,2:,1:-1]|
                padded[...,1:-1,:-2]|padded[...,1:-1,2:])
    return result


def dilate_review(mask):
    # Match the debug writer's OpenCV ellipse exactly.
    import cv2
    size=FINAL_MASK_KERNEL_SIZE
    footprint=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    source=mask.float()[None,None];padded=F.pad(source,(0,0,FINAL_MASK_RADIUS,FINAL_MASK_RADIUS));result=torch.zeros_like(source)
    h=mask.shape[0]
    for row in range(size):
        radius=int(footprint[row].sum())//2
        line=F.max_pool2d(padded[...,row:row+h,:],(1,2*radius+1),stride=1,padding=(0,radius))
        result=torch.maximum(result,line)
    return result[0,0]>0
