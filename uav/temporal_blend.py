"""Positive raised-cosine temporal window weights, normalized across predictions."""
import math


def window_ranges(length,window=10,overlap=5):
    starts=list(range(0,max(1,length-window+1),window-overlap))
    tail=max(0,length-window)
    if starts[-1]!=tail:starts.append(tail)
    return [(start,min(length,start+window)) for start in starts]


def window_weight(index,center,radius):
    distance=(index-center)/(radius+1)
    return .5*(1+math.cos(math.pi*distance))
