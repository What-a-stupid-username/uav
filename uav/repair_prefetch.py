"""Prepare exactly the production repair inputs on one bounded CPU worker."""
import cv2
import numpy as np


def prepare_repair(image,mask):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return rgb,repair_boxes(mask)


def repair_boxes(mask):
    """Group final-mask support for crop planning, without changing the mask."""
    # Region planning dilates only its support, never the final repair mask.
    grouped=cv2.dilate(mask.view(np.uint8),np.ones((31,31),np.uint8))
    contours,_=cv2.findContours(grouped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour);boxes.append([x,y,x+w,y+h])
    return boxes
