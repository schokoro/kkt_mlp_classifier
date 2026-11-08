"""S08 preparation and separately authorized single-use evaluation.

Report exit zero means report construction succeeded, not completed training or NI.
Production execute must follow independent tests/review and orchestration approval.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'plan'):
        p = subs.add_parser(name)
        p.add_argument('--output-root', required=True)
        p.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
        p.add_argument('--num-threads', type=int)
    for name in ('execute', 'evaluate', 'report'):
        p = subs.add_parser(name)
        p.add_argument('--output-root', required=True)
        if name == 'evaluate':
            p.add_argument('--approval-record', required=True)
            p.add_argument('--trusted-approval-sha256', required=True,
                           help='Trusted caller input ONLY after actual user consent; never auto-derive from payload')
            p.add_argument('--report-dir')
        if name == 'report':
            p.add_argument('--report-dir')
    p = subs.add_parser('rebuild')
    p.add_argument('--report-dir', required=True)
    p.add_argument('--output-html', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'rebuild':
            from kkt_clf.s08_report import render_report
            result = {'html': render_report(args.report_dir, args.output_html)}
        elif args.command == 'report':
            from kkt_clf.s08_report import build_report
            result = build_report(args.output_root, args.report_dir)
        else:
            from kkt_clf.s08 import S08Config, evaluate, execute, preflight, write_plan
            if args.command in ('execute', 'evaluate'):
                result = (execute(args.output_root) if args.command == 'execute' else
                          evaluate(args.output_root, args.approval_record, args.trusted_approval_sha256,
                                   report_dir=args.report_dir))
            else:
                cfg = S08Config(device=args.device, num_threads=args.num_threads)
                result = (preflight if args.command == 'preflight' else write_plan)(cfg, args.output_root)
                if args.command == 'plan':
                    result = {'output_root': args.output_root, 'runs': len(result['runs']),
                              'gate': 'awaiting_user_approval', 'delta_NI': None}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2 if result.get('passed') is False or result.get('failed') or result.get('stopped_reason') else 0
    except Exception as exc:
        print(f'S08 gate rejected: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
