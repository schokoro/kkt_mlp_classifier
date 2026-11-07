"""S07: preflight, immutable planning, single-use execution, report and table-only rebuild.

The report command exits zero for successful report construction, including a
partial diagnostic report. Study completion requires complete=true and all 36
statuses successful in its JSON output and saved manifest; exit zero alone does
not establish study success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'plan'):
        p = subs.add_parser(name)
        p.add_argument('--output-root', required=True, help='Fresh directory; never overwritten')
        p.add_argument('--device', default='cpu', choices=('cpu', 'cuda', 'auto'))
        p.add_argument('--num-threads', type=int, default=None)
    p = subs.add_parser('execute')
    p.add_argument('--output-root', required=True)
    p = subs.add_parser('report', help='Build a report; exit 0 means construction succeeded, not study completion',
                        description='Partial diagnostic reports also exit 0. Check complete=true and '
                        'statuses.success=36 in JSON/manifest for study completion.')
    p.add_argument('--output-root', required=True)
    p.add_argument('--report-dir', default=None, help='Fresh report directory')
    p = subs.add_parser('rebuild')
    p.add_argument('--report-dir', required=True, help='Only saved report tables and manifest needed')
    p.add_argument('--output-html', required=True, help='New HTML file')
    args = parser.parse_args()
    try:
        if args.command == 'rebuild':
            from kkt_clf.s07_report import render_report
            result = {'html': render_report(args.report_dir, args.output_html)}
        elif args.command == 'report':
            from kkt_clf.s07_report import build_report
            result = build_report(args.output_root, args.report_dir)
        else:
            from kkt_clf.s07 import S07Config, execute, preflight, write_plan
            if args.command == 'execute':
                result = execute(args.output_root)
            else:
                cfg = S07Config(device=args.device, num_threads=args.num_threads)
                result = (preflight if args.command == 'preflight' else write_plan)(cfg, args.output_root)
        # Compact CLI output; complete evidence lives on disk.
        if args.command == 'plan':
            result = {'output_root': args.output_root, 'runs': len(result['runs']),
                      'dependencies': len(result['dependencies'])}
        elif args.command == 'preflight':
            result = {'output_root': args.output_root, 'passed': result['passed'], 'error': result.get('error')}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if result.get('passed') is False or result.get('failed') or result.get('stopped_reason') else 0
    except Exception as exc:
        print(f'S07 gate rejected: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
