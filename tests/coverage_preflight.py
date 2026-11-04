"""Synthetic capture tests; no image/video files are read."""
import cv2
from uav.coverage_preflight import check_coverage,box_union_coverage,coverage_statistics,box_coverage_statistics
from uav import UAVConfig
from uav.cli import build_parser
class Capture:
    def __init__(self,source):self.released=False;self.index=0
    def isOpened(self):return True
    def get(self,key):return 12 if key==cv2.CAP_PROP_FRAME_COUNT else 24
    def set(self,key,value):self.index=value;return True
    def read(self):return True,self.index
    def release(self):self.released=True
captures=[]
def factory(source):
    c=Capture(source);captures.append(c);return c
a=check_coverage('synthetic',.2,123,lambda frame:dict(coverage=.25,single_coverage=.25),capture_factory=factory)
assert a['skipped'] and len(a['samples'])==4 and len({s['source_frame'] for s in a['samples']})==4
b=check_coverage('synthetic',.25,123,lambda frame:dict(coverage=.25,single_coverage=.25),capture_factory=factory)
assert not b['skipped'] and a['samples']==b['samples'] and all(c.released for c in captures)
try:check_coverage('synthetic',.2,123,lambda frame:dict(coverage=float('nan'),single_coverage=0),capture_factory=factory)
except ValueError:pass
else:raise AssertionError('invalid coverage accepted')
assert captures[-1].released
for value in (-.1,1.1,float('nan'),True):
    try:UAVConfig(max_watermark_coverage=value)
    except ValueError:pass
    else:raise AssertionError(value)
args=build_parser().parse_args(['--source','synthetic','--output','out','--max-watermark-coverage','.2'])
assert args.max_watermark_coverage==.2
assert box_union_coverage([[0,0,6,10],[4,0,10,10],[0,0,6,10]],10,10)==1
assert box_union_coverage([[-5,-5,5,5]],10,10)==.25
assert box_union_coverage([],10,10)==0
stats=coverage_statistics([0,.1,.2,.4])
assert stats['max_coverage']==.4 and stats['retained_frames']==2
assert abs(stats['filtered_mean_coverage']-.3)<1e-8
assert coverage_statistics([0,0,0,0])['no_detection']
c=check_coverage('synthetic',.5,123,lambda frame:dict(coverage=.8 if frame==a['samples'][0]['source_frame'] else 0,single_coverage=0),capture_factory=factory)
assert c['skipped'] and c['max_coverage']==.8 and c['mean_coverage']==.2
for value in (-.1,1.1,float('nan'),True):
    try:UAVConfig(max_single_watermark_coverage=value)
    except ValueError:pass
    else:raise AssertionError(value)
assert UAVConfig().max_single_watermark_coverage==.15
args=build_parser().parse_args(['--source','synthetic','--output','out','--max-single-watermark-coverage','.08'])
assert args.max_single_watermark_coverage==.08
measure=lambda frame:box_coverage_statistics([[0,0,4,10],[6,0,10,10]],10,10)
d=check_coverage('synthetic',.9,123,measure,single_threshold=.3,capture_factory=factory)
assert d['skipped'] and d['skip_reasons']==['single_watermark_coverage_exceeded']
assert d['max_single_coverage']==.4 and d['max_coverage']==.8
assert all(s['single_coverage']==.4 for s in d['samples'])
e=check_coverage('synthetic',None,123,measure,single_threshold=.4,capture_factory=factory)
assert not e['skipped'] # Strictly greater than the threshold.
f=check_coverage('synthetic',.7,123,measure,single_threshold=.3,capture_factory=factory)
assert f['skip_reasons']==['watermark_coverage_exceeded','single_watermark_coverage_exceeded']
assert box_coverage_statistics([[-5,-5,5,5]],10,10)==dict(coverage=.25,single_coverage=.25)
assert box_coverage_statistics([],10,10)==dict(coverage=0,single_coverage=0)
print('PASS: union/single coverage gates, reproducible sampling, clipping, cleanup, CLI validation')
