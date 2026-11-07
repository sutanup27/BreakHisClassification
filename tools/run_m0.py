"""Generate M0 evidence and Gate G0, or verify a frozen run without changing it."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from breakhis.research import records as r
from breakhis.research.runner import run_m0, verify_run, load_inputs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    try:
        r.validate_id(args.run_id)
        config=r.validate_m0_config(r.read_json(args.config),config_file=args.config)
        if not args.verify_only: load_inputs(args.config)
    except (ValueError,OSError) as error:
        parser.error(str(error))
    try:
        if args.verify_only:
            result=verify_run(r.safe_child(config['output_root'],args.run_id,must_exist=True))
        else:
            result=run_m0(args.config,args.run_id)
        print(json.dumps(result,indent=2))
        return result['exit_code']
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(json.dumps(dict(validated=False,error_type=type(error).__name__,error=str(error))),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
