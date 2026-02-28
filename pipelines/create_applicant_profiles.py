from pathlib import Path
import argparse
from dvclive import Live
import logging
from typing import Tuple, Mapping, Dict
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import polars as pl
from dotenv import load_dotenv
import os
from src.llm_response_cache import with_responses_cache
from openai import OpenAI
import random

from pipelines.utils import set_seed

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CLIENT = with_responses_cache(OpenAI(api_key=OPENAI_API_KEY))
OPENAI_MODEL_NAME = "gpt-5-2025-08-07"

SEED = 43

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def load_resumes(data_dir: Path) -> Mapping[str, str]:
    resumes_dir = data_dir / "resumes"
    resumes: Dict[str, str] = {}
    json_paths = list(resumes_dir.glob("*.json"))
    random.shuffle(json_paths)
    json_paths = json_paths[:5] #TODO: revert

    for json_path in json_paths:
        resume_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        resumes[resume_id] = obj["text"]
    return resumes


def load_rubrics(data_dir: Path, resume_ids: Tuple[str, ...]) -> Mapping[str, dict]:
    rubrics_dir = data_dir / "rubrics"
    rubrics: Dict[str, dict] = {}
    for json_path in rubrics_dir.glob("*.json"):
        rubric_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            rubrics[rubric_id] = json.load(f)

    resume_rubrics: Dict[str, dict] = {}
    for resume_id in resume_ids:
        domain = resume_id.split("_")[0]
        resume_rubrics[resume_id] = deepcopy(rubrics[domain])
    return resume_rubrics


def load_archetypes(data_dir: Path, resume_ids: Tuple[str, ...]) -> Mapping[str, Dict]:
    with (data_dir / "applicant_archetypes.json").open("r", encoding="utf-8") as f:
        domain_archetypes = json.load(f)

    resume_archetypes: Dict[str, dict] = {}
    for resume_id in resume_ids:
        domain = resume_id.split("_")[0]
        subset = {
            k: v for k, v in domain_archetypes[domain].items() if random.choice([True])
        }
        resume_archetypes[resume_id] = subset# TODO: REVERT deepcopy(domain_archetypes[domain])
    return resume_archetypes


def get_profile(resume_id: str, archetype_id: str, system_prompt: str, user_prompt) -> Tuple[str, str, str]:
    payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # response = OPENAI_CLIENT.responses.create(
    #     model=OPENAI_MODEL_NAME,
    #     input=payload,
    #     reasoning={"effort": "low"},
    # )

    return resume_id, archetype_id, "blank"


def create_profiles(
    exp: Live,
    data_dir: Path,
    save_path: Path,
    max_workers: int,
) -> Dict[Tuple[str, str], str]:
    resumes = load_resumes(data_dir)
    resume_ids = tuple(resumes.keys())
    rubrics = load_rubrics(data_dir, resume_ids)
    archetypes = load_archetypes(data_dir, resume_ids)

    system_prompt = \
"""
=== Task ===

You will be provided a Resume, Archetype and Rubric. Your task is to turn
the resume into a comprehensive profile that adheres to the Archetype based
on a strictly defined set of transformations. It is expected that the resultant
profile reads vastly differently to the Resume.

For each dimension in the Archetype, you will find LOW, MEDIUM and HIGH
descriptions in the RUBRIC which define the distribution of knowledge, skills 
and dispositions (KSD). The Archetype exists in a subspace of the Rubric
defined by the selected level for each of its dimensions.

=== Preprocessing === 

For EVERY resume item, invent the following missing information if absent:
  - To what degree was this individual involved?
  - What exactly did this individual do or contribute?
  - Why did this individual do what they did?
  - What does the subject understand about this experience?
  - What does the subject NOT understand about this experience?
  - List the KSD possessed by this individual relevant to the Rubric.

All of this information will subsequently transformed and aligned with the 
Archetype.

=== Archetype Adherence ===

For EACH dimension in the Archetype, you must apply the following:

If the given dimension level is LOW:
- Completely degrade all awards, accomplishments, experiences, complexity, metrics
  and skills.
- Remove remaining facts, details and general information that may imply
  a MEDIUM or HIGH level.
- Remove jargon and technical terms that an individual with a LOW level
  and minimal competency would not understand.
- (STRICT) Explicitly state all the KSD entailed by a MEDIUM level as lacking/absent
  wrt to the subject. Do not mention the fact that said lacking KSD belong to
  MEDIUM, simply declare that they do not belong to the subject.
- If the profile provides any inclination towards MEDIUM at all, you have FAILED.
- (STRICT) Do not mention positive behaviours - only mention negative behaviours and
  outcomes.
- 

Else if the given dimension level is MEDIUM:
- Reword all relevant Resume items to specifically sound like they belong
  to a MEDIUM level and not to a LOW or HIGH level.
- Remove all implications of a HIGH ability - prefer the lower end of MEDIUM
  than the HIGHER end.
- Ensure all projects, companies

Else if the given dimension level is HIGH:
- Invent ADDITIONAL details corresponding directly with what is expected of
  a HIGH level.
- Add specific end-to-end examples demonstrating HIGH competence.
- Add technical terms and jargon expected to be used by someone with HIGH
  competence.

(STRICT) Each transformation should only affect information relevant to
the given dimension and must not impact other dimensions.

=== Profile Format & Style ===

- 500-1000 words.
- Paragraph format.
- Unbiased (neutral sentiment and adjectives).
- Do not leak your task - the person reading the profile does not know about
  archetypes, rubrics and levels - they are purely for your inner workings.
- Always refer to the subject as the "subject"

=== Resume Completeness ===

(STRICT) You must ensure that every experience, project, tool, ability, hobby 
and award is included in the profile.

""".strip()

    user_prompt_tempalte = \
"""
<Resume>
{resume}
</Resume>

<Archetype>
{archetype}
</Archetype>

<Rubric>
{rubric}
</Rubric>
"""

    jobs = []
    for rid in resume_ids:
        for aid, archetype in archetypes[rid].items():
            user_prompt = user_prompt_tempalte.format(
                resume=resumes[rid],
                archetype=archetype,
                rubric=rubrics[rid],
            )
            jobs.append((
                rid,
                aid,
                system_prompt,
                user_prompt,
                resumes[rid],
                json.dumps(archetype, ensure_ascii=False),
            ))

    exp.log_param("max_workers", max_workers)
    exp.log_metric("num_jobs", len(jobs))

    out: Dict[Tuple[str, str], str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(get_profile, rid, aid, system_prompt, user_prompt): (rid, aid)
            for rid, aid, system_prompt, user_prompt, _, _ in jobs
        }
        for fut in as_completed(futures):
            rid, aid, profile = fut.result()
            out[(rid, aid)] = profile

    job_meta = {(rid, aid): (resume_text, archetype_json) for rid, aid, _, _, resume_text, archetype_json in jobs}

    rows = []
    for (rid, aid), profile in out.items():
        resume_text, archetype_json = job_meta[(rid, aid)]
        rows.append({
            "resume_id": rid,
            "archetype_id": aid,
            "resume": resume_text,
            "archetype_json": archetype_json,
            "profile": profile,
        })
    out = pl.DataFrame(rows)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(save_path)

    exp.log_metric("num_profiles", len(out))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--max-workers", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    save_path = Path(args.save_path).resolve()
    data_dir = Path(args.data_dir).resolve()

    exp_dir.parent.mkdir(parents=True, exist_ok=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with Live(dir=exp_dir) as exp_tracker:
        set_seed(SEED)
        exp_tracker.log_param("random_seed", SEED)

        create_profiles(
            exp=exp_tracker,
            data_dir=data_dir,
            save_path=save_path,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
