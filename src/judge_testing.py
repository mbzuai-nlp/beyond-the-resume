from src.judge import Judge, JudgeName, JUDGES, Judgement

from dataclasses import dataclass
import logging
from dvclive import Live
from pathlib import Path
from typing import Mapping, Dict, Tuple, Optional, Literal
from typing import get_args
import json
import polars as pl
from copy import deepcopy

# We define that a move > |4pp| defines a change. Therefore, a move of <= |4| is no-change.


# === Types ===


@dataclass(frozen=True)
class Config:
    test_name: str
    judge_name: JudgeName
    data_dir: Path
    save_path: Path


# === Globals ===


METAMORPHIC_TESTS = Literal[
    "irrelevance",
    "aggrandising",
    "repetition",
    "low-evidence",
    "medium-evidence",
    "high-evidence",
    "low-evidence-debase",
    "medium-evidence-debase",
    "high-evidence-debase",
]


# === Helpers ===


def _load_resumes(data_dir: Path) -> Mapping[str, str]:
    """Returns dict of (resume_id, resume_content)"""
    resumes_dir = data_dir / "resumes"
    out: Dict[str, str] = {}

    for json_path in resumes_dir.glob("*.json"):
        resume_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        out[resume_id] = obj["text"]

    return out


def _load_base_interviews(data_dir: Path) -> Mapping[str, Tuple[Mapping[str, str], ...]]:
    """Returns dict of (resume_id, base_interview_content)"""
    base_interviews_dir = data_dir / "judge-tests" / "base-interviews"
    out: Dict[str, Tuple[Mapping[str, str], ...]] = {}

    for json_path in base_interviews_dir.glob("*.json"):
        resume_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            turns = json.load(f)
        out[resume_id] = tuple(turns)

    return out


def _load_rubrics(data_dir: Path) -> Mapping[str, dict]:
    """Returns dict of (rubric_id, rubric)"""
    rubrics_dir = data_dir / "rubrics"
    out: Dict[str, dict] = {}

    for json_path in rubrics_dir.glob("*.json"):
        rubric_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            out[rubric_id] = json.load(f)

    return out


def _log_pc_passing_by_domain(
    exp: Live,
    scored_cases: list[tuple[str, bool]],
) -> None:
    """Logs metrics like pc_passing_sales, pc_passing_finance, ..."""
    per_domain: Dict[str, list[bool]] = {}

    for resume_id, passed in scored_cases:
        domain = resume_id.split("_")[0]
        per_domain.setdefault(domain, []).append(bool(passed))

    for domain, vals in per_domain.items():
        exp.log_metric(
            f"pc_passing_{domain}",
            sum(vals) / len(vals) if vals else 0.0,
        )


def _log_pc_passing_by_mode(
    exp: Live,
    scored_cases: list[tuple[str, bool]],
    modes_by_resume_id: Mapping[str, Optional[str]],
) -> None:
    """Logs metrics like pc_passing_xyz where xyz is the debase test mode."""
    per_mode: Dict[str, list[bool]] = {}

    for resume_id, passed in scored_cases:
        mode = modes_by_resume_id.get(resume_id)
        if mode is None:
            continue
        per_mode.setdefault(mode, []).append(bool(passed))

    for mode, vals in per_mode.items():
        exp.log_metric(
            f"pc_passing_{mode}",
            sum(vals) / len(vals) if vals else 0.0,
        )


def _load_metamorphic_test(
    data_dir: Path,
    test_name: str,
) -> Tuple[
    Mapping[str, Tuple[Mapping[str, str], ...]],
    Mapping[str, Optional[dict]],
    Mapping[str, Optional[str]],
]:
    turns: Dict[str, Tuple[Mapping[str, str], ...]] = {}
    belief_updates: Dict[str, Optional[dict]] = {}
    modes: Dict[str, Optional[str]] = {}

    if test_name.endswith("-debase"):
        injection_test_name = test_name[: -len("-debase")]

        with open(data_dir / "judge-tests" / f"{injection_test_name}.json") as f:
            injection_tests = json.load(f)["tests"]

        with open(data_dir / "judge-tests" / f"{test_name}.json") as f:
            debase_tests = json.load(f)["tests"]

        for resume_id, d in debase_tests.items():
            inj = injection_tests[resume_id]
            turns[resume_id] = (
                {"role": "interviewer", "message": inj["interviewer_turn"]},
                {"role": "applicant", "message": inj["applicant_turn"]},
                {"role": "interviewer", "message": d["interviewer_turn"]},
                {"role": "applicant", "message": d["applicant_turn"]},
            )
            belief_updates[resume_id] = d.get("belief_update")
            modes[resume_id] = d["mode"]

        return turns, belief_updates, modes

    with open(data_dir / "judge-tests" / f"{test_name}.json") as f:
        tests = json.load(f)["tests"]

    for resume_id, d in tests.items():
        turns[resume_id] = (
            {"role": "interviewer", "message": d["interviewer_turn"]},
            {"role": "applicant", "message": d["applicant_turn"]},
        )
        belief_updates[resume_id] = d.get("belief_update")
        modes[resume_id] = None

    return turns, belief_updates, modes


def _belief_updated_expectedly(
    prior_judgement: Judgement,
    post_judgement: Judgement,
    expected_belief_update: Optional[dict]
) -> bool:
    th = 0.04

    if expected_belief_update:
        target_dim = expected_belief_update["dimension"]
        target_level = expected_belief_update["level"]
        target_direction = expected_belief_update["direction"]

        prior_belief = prior_judgement.posteriors[target_dim][target_level]
        post_belief = post_judgement.posteriors[target_dim][target_level]

        delta = post_belief - prior_belief

        if target_direction == "down" and delta >= -th:
            return False
        if target_direction == "up" and delta <= th:
            return False
    else:
        for dim in prior_judgement.posteriors.keys():
            for level in ("low", "medium", "high"):
                prior_belief = prior_judgement.posteriors[dim][level]
                post_belief = post_judgement.posteriors[dim][level]

                if abs(prior_belief - post_belief) > th:
                    return False

    return True


def _is_close_to_uniform_prior(
    judgement: Judgement,
    dimension: str,
) -> Tuple[bool, float]:
    target = 1.0 / 3.0
    probs = judgement.posteriors[dimension]

    diffs = [
        abs(float(probs["low"]) - target),
        abs(float(probs["medium"]) - target),
        abs(float(probs["high"]) - target),
    ]
    max_abs_diff = max(diffs)
    return (max_abs_diff <= 0.04), max_abs_diff


def _applicant_step_transcripts(
    turns: Tuple[Mapping[str, str], ...],
) -> list[Tuple[Mapping[str, str], ...]]:
    out: list[Tuple[Mapping[str, str], ...]] = [tuple()]
    acc: list[Mapping[str, str]] = []
    n_app = 0
    for t in turns:
        acc.append(t)
        if t["role"] == "applicant":
            n_app += 1
            out.append(tuple(acc))
    return out


def _run_incremental(
    j: Judge,
    resumes: Mapping[str, str],
    rubrics: Mapping[str, dict],
    full_turns: Mapping[str, Tuple[Mapping[str, str], ...]],
    **judge_kwargs,
) -> Dict[str, list[Judgement]]:
    step_transcripts = {rid: _applicant_step_transcripts(full_turns[rid]) for rid in resumes}
    max_steps = max(len(v) for v in step_transcripts.values())

    per_rid: Dict[str, list[Judgement]] = {rid: [] for rid in resumes}

    for t in range(max_steps):
        batch_rids = [rid for rid in resumes if t < len(step_transcripts[rid])]
        batch_resumes = {rid: resumes[rid] for rid in batch_rids}
        batch_rubrics = {rid: rubrics[rid] for rid in batch_rids}
        batch_interviews = {rid: step_transcripts[rid][t] for rid in batch_rids}

        out = j.judge_next_turn(
            resumes=batch_resumes,
            interviews=batch_interviews,
            rubrics=batch_rubrics,
            **judge_kwargs,
        )

        for rid in batch_rids:
            per_rid[rid].append(out[rid])

    return per_rid


# == Public Functions ===


def run_judge_test(config: Config, exp: Live, **judge_kwargs):
    judge = JUDGES[config.judge_name]

    resumes = _load_resumes(config.data_dir)
    resume_ids = tuple(resumes.keys())
    base_interviews = _load_base_interviews(config.data_dir)
    base_rubrics = _load_rubrics(config.data_dir)

    if config.test_name in set(get_args(METAMORPHIC_TESTS)):
        rubrics = {}
        for resume_id in resume_ids:
            domain = resume_id.split("_")[0]
            rubrics[resume_id] = deepcopy(base_rubrics[domain])

        test_turns, expected_belief_updates, modes_by_resume_id = _load_metamorphic_test(
            config.data_dir, config.test_name
        )

        rows: list[dict] = []

        # Without base interviews: resume-only -> ... -> full test
        with judge(exp=exp) as j:
            full_turns = {rid: test_turns[rid] for rid in resume_ids}
            traj = _run_incremental(j, resumes, rubrics, full_turns, **judge_kwargs)

            without_base_interview_scores = []
            for rid in resume_ids:
                post_idx = len(traj[rid]) - 1
                prior_idx = post_idx - 1
                prior_j = traj[rid][prior_idx]
                post_j = traj[rid][post_idx]

                passed = _belief_updated_expectedly(
                    prior_j,
                    post_j,
                    expected_belief_updates.get(rid),
                )
                without_base_interview_scores.append(passed)

                exp_upd = expected_belief_updates[rid] or {}
                rows.append(
                    {
                        "resume_id": rid,
                        "condition": "without_base",
                        "passed": bool(passed),
                        "expected_dimension": exp_upd.get("dimension"),
                        "expected_level": exp_upd.get("level"),
                        "prior_posteriors_json": json.dumps(prior_j.posteriors),
                        "post_posteriors_json": json.dumps(post_j.posteriors),
                        "prior_justifications_json": json.dumps(prior_j.justifications),
                        "post_justifications_json": json.dumps(post_j.justifications),
                    }
                )

        # With base interviews: resume-only -> ... -> base -> ... -> full test
        with judge(exp=exp) as j:
            full_turns = {rid: (*base_interviews[rid], *test_turns[rid]) for rid in resume_ids}
            traj = _run_incremental(j, resumes, rubrics, full_turns, **judge_kwargs)

            with_base_interview_scores = []
            for rid in resume_ids:
                post_idx = len(traj[rid]) - 1
                prior_idx = post_idx - 1
                prior_j = traj[rid][prior_idx]
                post_j = traj[rid][post_idx]

                passed = _belief_updated_expectedly(
                    prior_j,
                    post_j,
                    expected_belief_updates.get(rid),
                )
                with_base_interview_scores.append(passed)

                exp_upd = expected_belief_updates[rid] or {}
                rows.append(
                    {
                        "resume_id": rid,
                        "condition": "with_base",
                        "passed": bool(passed),
                        "expected_dimension": exp_upd.get("dimension"),
                        "expected_level": exp_upd.get("level"),
                        "prior_posteriors_json": json.dumps(prior_j.posteriors),
                        "post_posteriors_json": json.dumps(post_j.posteriors),
                        "prior_justifications_json": json.dumps(prior_j.justifications),
                        "post_justifications_json": json.dumps(post_j.justifications),
                    }
                )

        num_resumes = len(resume_ids)
        exp.log_param("num_resumes", num_resumes)
        exp.log_metric(
            "pc_passing_without_base_interviews",
            sum(without_base_interview_scores) / num_resumes,
        )
        exp.log_metric(
            "pc_passing_with_base_interviews",
            sum(with_base_interview_scores) / num_resumes,
        )
        exp.log_metric(
            "pc_passing",
            (sum(without_base_interview_scores) + sum(with_base_interview_scores)) / (2 * num_resumes),
        )

        combined_scores = (
            list(zip(resume_ids, without_base_interview_scores))
            + list(zip(resume_ids, with_base_interview_scores))
        )

        _log_pc_passing_by_domain(exp, combined_scores)

        if config.test_name.endswith("-debase"):
            _log_pc_passing_by_mode(exp, combined_scores, modes_by_resume_id)

        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_parquet(config.save_path)
        return
    elif config.test_name == "uniform-prior":
        with open(config.data_dir / "judge-tests" / "uniform-prior.json") as f:
            test_specs = json.load(f)

        # Add test rubric dimension to base domain rubric
        new_rubrics = {}
        test_dimensions = {}
        for resume_id in resume_ids:
            domain = resume_id.split("_")[0]
            base_rubric = base_rubrics[domain]

            test_rubric = base_rubrics[test_specs[resume_id]["rubric"]]
            test_dimension_name = test_specs[resume_id]["dimension"]

            new_rubric = deepcopy(base_rubric)
            new_rubric[test_dimension_name] = test_rubric[test_dimension_name]

            new_rubrics[resume_id] = new_rubric
            test_dimensions[resume_id] = test_dimension_name
        
        empty_interviews: Dict[str, Tuple[Mapping[str, str], ...]] = {
            rid: tuple() for rid in resume_ids
        }

        with judge(exp=exp) as j:
            judgements = j.judge_next_turn(
                resumes=resumes,
                interviews=empty_interviews,
                rubrics=new_rubrics,
                **judge_kwargs,
            )

        rows: list[dict] = []
        passing: list[bool] = []
        for rid in resume_ids:
            dim = test_dimensions[rid]
            passed, max_abs_diff = _is_close_to_uniform_prior(judgements[rid], dim)
            passing.append(bool(passed))

            post = judgements[rid].posteriors[dim]
            p_low = float(post["low"])
            p_med = float(post["medium"])
            p_high = float(post["high"])

            rows.append(
                {
                    "resume_id": rid,
                    "passed": bool(passed),
                    "test_dimension": dim,
                    "p_low": p_low,
                    "p_medium": p_med,
                    "p_high": p_high,
                    "max_abs_diff_from_uniform": float(max_abs_diff),
                    "posteriors_json": json.dumps(judgements[rid].posteriors),
                    "justifications_json": json.dumps(judgements[rid].justifications),
                }
            )

        num_resumes = len(resume_ids)
        exp.log_param("num_resumes", num_resumes)

        exp.log_metric(
            "pc_passing",
            sum(passing) / num_resumes if num_resumes else 0.0,
        )

        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(rows)
        df.write_parquet(config.save_path)

        