"""Explicit parameters, no legacy env fallback, and run-context isolation."""
from dataclasses import fields
import os,subprocess,sys
from uav import UAVConfig
from uav.config import get_config,use_config
from uav.cli import build_parser
parser=build_parser()
def parsed(*args):
 values=vars(parser.parse_args(['--source','input.mp4','--output','output',*args]));values.pop('source');values.pop('output');values.pop('worker');return UAVConfig(**values)
a=parsed();assert a==UAVConfig()
assert a.block_overlap==3 and a.segment_overlap==1
assert a.propainter_reference_host_cache_mib==4096
assert parsed('--propainter-reference-host-cache-mib','0').propainter_reference_host_cache_mib==0
assert a.repair and a.sift_backend=="cuda" and a.mask_pixel_threshold==.23 and a.mask_dilation==4
assert a.raft_scale==a.flow_completion_scale==.5
assert a.raft_iters==4
assert a.repair_scale==.7 and parsed('--repair-scale','1').repair_scale==1
quality=parsed('--high-quality')
assert quality==UAVConfig.high_quality()
assert (quality.block_overlap,quality.repair_scale,quality.raft_iters)==(5,1,8)
assert parsed('--high-quality','--raft-iters','4').raft_iters==4
assert UAVConfig.high_quality(raft_iters=4).raft_iters==4
assert parsed('--raft-iters','8').raft_iters==8
assert a.sam3_global_resolution==1008 and not hasattr(a,"two_pass")
b=parsed('--raft-scale','1','--flow-completion-scale','1','--no-compile','--global-stride','4','--block-overlap','3')
assert b.raft_scale==b.flow_completion_scale==1 and not b.sam3_compile and b.sam3_global_stride==4
with use_config(b):
 assert get_config()==b
 with use_config(a):assert get_config()==a
 assert get_config()==b
assert get_config()==a
try:
 with use_config(b):raise RuntimeError('check restoration')
except RuntimeError:pass
assert get_config()==a
for kw in ({'mask_pixel_threshold':0},{'mask_pixel_threshold':1},{'mask_dilation':-1},{'raft_scale':0},{'flow_completion_scale':float('nan')},{'fps':0},{'reference_seconds':float('inf')},{'sam3_global_stride':7},{'block_overlap':2},{'random_start':True}):
 try:UAVConfig(**kw)
 except ValueError:pass
 else:raise AssertionError(kw)
# Check every public config field can be represented on CLI.
for field in fields(a):
 if field.name in ('source','output'):continue
 value=getattr(a,field.name);flag='--'+field.name.replace('_','-')
 if isinstance(value,bool):
  extra=['--max-seconds','15'] if field.name=='random_start' else []
  assert getattr(parsed(flag,*extra),field.name) is True and getattr(parsed('--no-'+field.name.replace('_','-'),*extra),field.name) is False
 elif value is not None:assert getattr(parsed(flag,str(value)),field.name)==value
legacy=dict(os.environ,SAM3_GLOBAL_RESOLUTION='504',SAM3_COMPILE='0',PROPAINTER_RAFT_SCALE='1',PROPAINTER_FLOW_COMPLETION_SCALE='1')
subprocess.run([sys.executable,'-c','from uav import UAVConfig; from uav.config import get_config; assert get_config()==UAVConfig(); assert get_config().raft_scale==0.5'],env=legacy,check=True)
print('PASS: CLI/Python defaults, all options, invalid values, context restoration and ignored legacy env')
