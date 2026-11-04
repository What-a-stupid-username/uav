"""UAV — Unmark Any Video. Explicit SAM3 and ProPainter inference parameters."""
import argparse
from dataclasses import fields
from pathlib import Path
from .config import UAVConfig


class _HighQuality(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for name in ('block_overlap','repair_scale','raft_iters'):
            setattr(namespace,name,getattr(UAVConfig.high_quality(),name))


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--worker',action='store_true',help='wait for JSON-lines jobs on stdin; retain model weights')
    parser.add_argument('--source',type=Path,action='append',nargs='+',
                        help='one or more video paths; may be repeated')
    parser.add_argument('--output',type=Path)
    defaults=UAVConfig()
    parser.add_argument('--high-quality',action=_HighQuality,nargs=0,default=argparse.SUPPRESS,
                        help='5-frame overlap, original repair resolution, 8 RAFT iterations; later flags override')
    optional_types={'max_seconds':float,'max_watermark_coverage':float,'max_single_watermark_coverage':float}
    for field in fields(defaults):
        if field.name in ('source','output'):continue
        default=getattr(defaults,field.name)
        flags=['--'+field.name.replace('_','-')]
        if field.name.startswith('sam3_'):flags.insert(0,'--'+field.name[5:].replace('_','-'))
        if field.name.startswith('propainter_'):flags.insert(0,'--'+field.name[11:].replace('_','-'))
        kwargs=dict(dest=field.name,default=default,help=field.name.replace('_',' '))
        if field.name in ('max_watermark_coverage','max_single_watermark_coverage'):
            kwargs['help']='startup coverage limit (0–1), sampled over entire source; 1 disables this gate'
        if isinstance(default,bool):kwargs['action']=argparse.BooleanOptionalAction
        else:kwargs['type']=optional_types.get(field.name,type(default))
        parser.add_argument(*flags,**kwargs)
    return parser


def main(argv=None):
    parser=build_parser();args=vars(parser.parse_args(argv))
    worker=args.pop('worker')
    sources=[source for group in (args.pop('source') or []) for source in group];output=args.pop('output')
    try:config=UAVConfig(**args)
    except (TypeError,ValueError) as error:parser.error(str(error))
    if worker:
        if sources or output is not None:parser.error('--worker receives source/output through stdin')
        from .worker import serve
        serve(config)
        return
    if not sources or output is None:parser.error('--source and --output are required unless using --worker')
    for source in sources:
        if not source.is_file():parser.error(f'source file does not exist: {source}')
    from .api import run, run_batch
    if len(sources)==1:run(sources[0],output,config)
    else:run_batch(sources,output,config)


if __name__=='__main__':main()
