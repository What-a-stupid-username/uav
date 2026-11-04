"""Sequential regression and additive publication to an existing static review page."""
import argparse
import html
import json
from pathlib import Path
import re


def processing_rates(report):
    """Skipped/failed jobs cannot masquerade as fast successful processing."""
    if not report.get('complete') or report.get('skipped'):
        return None
    frames = report.get('rendered_frames', 0)
    if not frames or frames != report.get('frames'):
        return None
    timing = report.get('timing', {})
    algorithm = timing.get('first_pass_seconds', 0)+timing.get('second_pass_seconds', 0)
    total = report.get('total_seconds', 0)
    return dict(algorithm_fps=frames/algorithm if algorithm else None,
                overall_fps=frames/total if total else None)


def publish(root, run_name, rows):
    root = Path(root)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
        raise ValueError('run name must be a single safe path component')
    begin, end = f'<!-- TEMPLATE_MASK_{run_name}_START -->', f'<!-- TEMPLATE_MASK_{run_name}_END -->'
    repairing = any(row.get('report', {}).get('repair') for row in rows)
    title = ('新模板方案 + ProPainter：完整修复回归' if repairing else
             '共享模板实验：旧方案 / 新方案第一遍与最终 mask')
    pieces = [begin, f'<section id="template-mask-{run_name}"><h1>'+title+'</h1>',
        '<p>批次：'+html.escape(run_name)+'</p>',
        '<p>生产 CLI 使用共享模板 mask 引擎。24 fps、全局每 8 帧。',
        '待人工判断质量；程序只统计数值。模板颜色含义见各批次说明。',
        '绿色为普通覆盖，橙色为保守覆盖贡献，画面注明原因和额外半径。',
        '跳过视频仅提供诊断，不是可交给修复的最终输出。完成但有警告不代表失败。</p>']
    pieces += ['<table><thead><tr><th>视频</th><th>状态</th><th>两遍 fps</th>',
               '<th>整体 fps（含审核）</th><th>警告区间数</th><th>跳过／错误原因</th></tr></thead><tbody>']
    for row in rows:
        report = row.get('report', {})
        rates = processing_rates(report)
        rate = lambda key: f'{rates[key]:.2f}' if rates and rates[key] else '—'
        status = dict(queued='排队', running='运行中', complete='完成', skipped='跳过', error='运行错误',
                      interrupted='已中止（旧实验）', superseded='已由新批次替代').get(row['status'], row['status'])
        if row['status'] == 'complete' and report.get('warnings'):
            status += '（有警告）'
        reason = report.get('reason') or report.get('error') or row.get('error') or '—'
        pieces += [f'<tr><td>{row["index"]}</td><td>{html.escape(status)}</td>',
            f'<td>{rate("algorithm_fps")}</td><td>{rate("overall_fps")}</td>',
            f'<td>{len(report.get("warnings", []))}</td><td>{html.escape(reason)}</td></tr>']
    pieces += ['</tbody></table>']
    for row in rows:
        index = row['index']; directory = f'{run_name}/video-{index:02d}'
        pieces += [f'<article><h2>视频 {index} · {html.escape(row["status"])}</h2>',
                   '<p>'+html.escape(row['name'])+f' · 源帧起点 {row["start"]}</p>']
        report = row.get('report', {})
        if row.get('error'):
            pieces += ['<p>运行错误：'+html.escape(row['error'])+'</p>']
        if report:
            stats = report.get('stats', {})
            config = report.get('config', {})
            if report.get('evidence_reuse'):
                pieces += ['<p>本次复用了上一轮全局检测和原图特征缓存，耗时不代表从零运行速度。</p>']
            pieces += ['<p>Stage 1 使用全局粗概率图关联和追踪；Stage 2 对共享模板选最多 8 份原图精修。',
                '正方形边长=max(目标长边×1.7，视频短边×35%)，统一 1008 送检。',
                '精修失败回退粗模板，外扩半径加倍。</p>']
            if report.get('rendering', {}).get('object_bounds') == 'valid_mask_support':
                pieces += ['<p>对象框修正：范围只由有效笔画概率超过 0.23 的像素扩展；',
                    '概率图存储留白、低概率边缘和 SAM 背景裁剪不计入对象尺寸。</p>']
            geometry = report.get('geometry_validation')
            if geometry:
                pieces += ['<p>'+html.escape(geometry['summary'])+'</p>',
                    f'<p><a href="{directory}/geometry-validation.json">各次闪现的框尺寸与尺度数值</a></p>']
            pieces += ['<p>'+html.escape(str({k:stats.get(k) for k in ('templates', 'instances', 'observations', 'template_calls')}))+
                       f' · {report.get("total_seconds", 0):.1f} 秒 · '+html.escape(str(report.get('reason', '')) )+'</p>']
            pieces += [f'<p><a href="{directory}/report.json">数值报告</a> · <a href="{directory}/warnings.jsonl">不确定区间</a></p>']
            if report.get('pending'):
                pieces += ['<p>重新归属 '+
                    str(stats.get('pending_resolved', 0))+' 份观测；仍属疑似模板 '+str(report['pending']['unresolved'])+' 份。</p>']
            if report.get('templates'):
                pieces += ['<p>渐进模板：'+str(sum(not t.get('provisional', True) for t in report['templates']))+
                    ' 个已由多份原图确认，'+str(sum(t.get('provisional', True) for t in report['templates']))+
                    ' 个仍属疑似；未关联旧模板的独立模板也参与第二遍输出。</p>']
            stages = report.get('stage_seconds', {})
            if stages:
                pieces += ['<p>阶段耗时（秒）：'+html.escape(str({k:round(v, 2) for k, v in stages.items()}))+'</p>']
            if report.get('review_error'):
                pieces += ['<p>审核视频生成错误：'+html.escape(report['review_error'])+'</p>']
        repair = report.get('repair',{})
        if repair.get('complete'):
            pieces += ['<h3>ProPainter 修复结果</h3>',
                f'<video controls playsinline preload="none" src="{directory}/repaired.mp4"></video>',
                f'<p><a href="{directory}/repaired.mp4" target="_blank">单独打开修复视频</a></p>']
        review = report.get('review', {})
        review_name = review.get('path', 'mask-review.mp4')
        if Path(review_name).name != review_name:
            raise ValueError('review path must be a filename')
        if report.get('complete') and (root/directory/review_name).is_file():
            first_label, final_label = review.get('first', '左侧'), review.get('final', '右侧')
            dilation_text = '精细模板外扩 4–12 原图像素，粗 mask 回退时加倍至 8–24 像素'
            pieces += ['<h3>本批次新实现：请审核'+final_label+'的最终 mask</h3>',
                '<p>'+first_label+'＝新实现第一遍（不外扩）；'+final_label+'＝新实现最终 mask（'+dilation_text+'，画面标注实际半径）。',
                '两个画面都属于本批次新实现。跳过时最终画面仅为诊断，不可交给修复。</p>']
            if review.get('native_width'):
                pieces += ['<p>审核画质：每个画面保留原始 '+str(review['native_width'])+'×'+
                    str(review['native_height'])+' 像素，框线 1 像素，不缩小后再编码。',
                    '横视频上下排列，竖视频左右排列。可打开原文件或全屏检查细节。</p>']
            pieces += [f'<video controls playsinline preload="none" src="{directory}/{review_name}"></video>',
                f'<p><a href="{directory}/{review_name}" target="_blank">单独打开新实现审核视频</a></p>']
            if review.get('html'):
                pieces += [f'<p><a href="{directory}/{review["html"]}">打开完整 debug 页面（两阶段与全部模板）</a></p>']
            old = f'latest21-20261022/video-{index:02d}/mask-passes.mp4'
            if (root/old).is_file():
                pieces += ['<details><summary>旧 UAV 基线对照：右半边为旧实现最终 mask（不是本次新结果）</summary>',
                           '<p>本行左侧＝旧 UAV 第一遍原始 mask；本行右侧＝旧 UAV 概率精修与外扩后的最终 mask。',
                           '本行两侧都属于旧实现；与上方新实现使用同一源帧起点。比较最终质量时，请比较新实现的'+final_label+'和旧实现的右半边。</p>',
                           f'<video controls playsinline preload="none" src="{old}"></video></details>']
            template_path = root/directory/'templates.json'
            if template_path.is_file():
                pieces += ['<details><summary>模板与原始观测来源</summary>']
                for template in json.loads(template_path.read_text()):
                    if not template.get('path'):
                        pieces += ['<p>模板 '+str(template['id'])+'：'+html.escape(template.get('reason', '无可用原图'))+'</p>']
                        continue
                    sources = ', '.join(str(s['source_frame']) for s in template['sources'])
                    fallback_label = ' · 粗 mask 回退' if template.get('coarse_fallback') else ''
                    pieces += [f'<p>模板 {template["id"]}{fallback_label} · 原始帧 {sources}</p>',
                               f'<img loading="lazy" style="max-width:100%" src="{directory}/{template["path"]}">']
                pieces += ['</details>']
        pieces += ['</article>']
    pieces += ['</section>'+end]
    page = root/'index.html'
    text = page.read_text() if page.exists() else '<!doctype html><meta charset="utf-8"><title>Mask review</title>'
    if begin in text:
        left = text.index(begin); right = text.index(end, left)+len(end)
        text = text[:left]+text[right:]
    insert = text.find('</style>')
    insert = insert+len('</style>') if insert >= 0 else text.find('<section')
    if insert < 0:
        insert = len(text)
    temporary = page.with_suffix('.'+run_name+'.tmp')
    temporary.write_text(text[:insert]+''.join(pieces)+text[insert:])
    temporary.replace(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='existing latest21 status.json (metadata only)')
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--run-name', default='template-evaluation')
    parser.add_argument('--indices', default='4,6,12,14,21')
    parser.add_argument('--max-seconds', type=float, default=15.)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('run-name must be a single safe path component')
    root = Path(args.output_root)
    selected = [int(x) for x in args.indices.split(',')]
    catalog = {r['index']:r for r in json.loads(Path(args.manifest).read_text())['jobs']}
    rows = [dict(index=i, name=catalog[i]['name'], start=catalog[i]['expected_start'], status='queued') for i in selected]
    destination = root/args.run_name; destination.mkdir(parents=True, exist_ok=True)
    def save_status():
        temporary = destination/'status.tmp'
        temporary.write_text(json.dumps(rows, indent=2))
        temporary.replace(destination/'status.json')
    save_status()
    publish(root, args.run_name, rows)
    from .runner import run
    from .config import TemplateConfig
    import torch
    from uav.sam3_model import build
    from uav.inference_weights import compact_weights
    torch.set_num_threads(4)
    with torch.inference_mode():
        model = build(); compact_weights(model.backbone)
    for row in rows:
        folder = destination/f'video-{row["index"]:02d}'
        if (folder/'report.json').is_file():
            report = json.loads((folder/'report.json').read_text())
            if report.get('complete'):
                row.update(status='skipped' if report.get('skipped') else 'complete', report=report)
                publish(root, args.run_name, rows)
                continue
            raise RuntimeError(f'incomplete prior run: use a new run name ({folder})')
        row['status'] = 'running'
        save_status()
        publish(root, args.run_name, rows)
        print(json.dumps(dict(event='regression_start', index=row['index'], name=row['name'])), flush=True)
        try:
            report = run(Path(args.input_root)/row['name'], folder, TemplateConfig(),
                max_seconds=args.max_seconds, start_source_frame=row['start'], model=model)
            row.update(status='skipped' if report['skipped'] else 'complete', report=report)
        except Exception as error:
            row.update(status='error', error=type(error).__name__+': '+str(error))
            print(json.dumps(dict(event='regression_error', **{k:v for k,v in row.items() if k != 'report'})), flush=True)
        save_status()
        publish(root, args.run_name, rows)
        print(json.dumps(dict(event='regression_done', index=row['index'], status=row['status'])), flush=True)


if __name__ == '__main__':
    main()
