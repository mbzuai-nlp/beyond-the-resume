from src.simulation import run_interview_simulations, Config
from pipelines.utils import set_seed

from pathlib import Path
import argparse
from dvclive import Live
import logging

SEED = 43

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True
)

def parse_kv(s: str):
    # key=value
    if "=" not in s:
        raise argparse.ArgumentTypeError("Expected KEY=VALUE")
    k, v = s.split("=", 1)
    return k, v

def coerce_value(v: str):
    low = v.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interviewer-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--judge-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
    )
    parser.add_argument("--kw", action="append", type=parse_kv, default=[],
                   help="Extra judge kwargs as KEY=VALUE. Repeatable.")
    return parser.parse_args()

def main():
    args = parse_args()
    interviewer_name = args.interviewer_name
    judge_name = args.judge_name
    exp_dir = Path(args.exp_dir).resolve()
    save_path = Path(args.save_path).resolve()
    data_dir = Path(args.data_dir).resolve()

    judge_kwargs = {k: coerce_value(v) for k, v in args.kw}
    
    exp_dir.parent.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config = Config(
        interviewer_name=interviewer_name,
        judge_name=judge_name,
        data_dir=data_dir,
        save_path=save_path
    )

    with Live(dir=exp_dir) as exp_tracker:
        set_seed(SEED)
        # Log config
        exp_tracker.log_param("random_seed", SEED)
        for k, v in (config.__dict__ | judge_kwargs).items():
            exp_tracker.log_param(k, str(v))

        run_interview_simulations(config, exp_tracker, **judge_kwargs)


if __name__ == "__main__":
    main()
