"""Profile the public pipeline without review media or intermediate mask files.

Run inside the UAV environment. Source decoding requires explicit media access
permission. The script emits numeric JSON only; it never opens generated media.
"""
import argparse
import cProfile
import json
from pathlib import Path
import pstats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=15.)
    parser.add_argument('--start-source-frame',type=int,default=0)
    parser.add_argument('--no-repair',action='store_true')
    parser.add_argument('--sam3-trt-engine', default='')
    parser.add_argument('--mask-reference-limit', type=int, default=8)
    parser.add_argument('--cpu-profile',action='store_true',
                        help='add main-thread cProfile; CPU timings include profiler overhead')
    args=parser.parse_args()
    from uav import UAVConfig,run
    profile=cProfile.Profile() if args.cpu_profile else None
    config=UAVConfig(max_seconds=args.max_seconds,
                     start_source_frame=args.start_source_frame,repair=not args.no_repair,
                     sam3_trt_engine=args.sam3_trt_engine,mask_reference_limit=args.mask_reference_limit)
    if profile:profile.enable()
    try:
        report=run(args.source,args.output,config)
    finally:
        if profile:profile.disable()
    result=report['runs'][0]
    # Retain the existing single-segment detail and expose video-level bounds.
    video_summary={key:report.get(key) for key in ('frames','output_frames','total_seconds','segmentation','frame_memory')}
    summary={key:result.get(key) for key in ('complete','skipped','frames','rendered_frames',
        'total_seconds','stage_seconds','timing','stage2_phases','sam','sam_stages',
        'discovery_prefetch','matching','frame_memory','peak_rss_mib')}
    summary['video']=video_summary
    repair=result.get('repair',{})
    summary['repair']={key:repair.get(key) for key in ('output_frames','repair_host_seconds',
        'stages','final_decode_seconds','output','flow_cache','reference_bank',
        'transformer_calls','pending_features')}
    summary['profiling']={'main_thread_cprofile':bool(profile),
                          'note':'CPU profile times include CUDA waits and nested calls; do not sum inclusive times.'}
    if profile:
        rows=[]
        for (filename,line,function),(primitive,calls,own,cumulative,callers) in pstats.Stats(profile).stats.items():
            rows.append(dict(file=filename,line=line,function=function,calls=calls,
                             self_seconds=own,cumulative_seconds=cumulative))
        summary['cpu_self']=sorted(rows,key=lambda r:r['self_seconds'],reverse=True)[:40]
        summary['cpu_cumulative']=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:40]
    (args.output/'profile-summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
