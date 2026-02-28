from __future__ import annotations

from typing import Tuple, Mapping, Literal, Optional, Any, List, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

from dotenv import load_dotenv
from dvclive import Live
from openai import OpenAI

from src.llm_response_cache import with_responses_cache


JudgeName = Literal["independent", "previous-judgement-aware"]
UnitInterval = float  # [0, 1]
JudgementLevel = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Judgement:
    posteriors: Mapping[str, Mapping[JudgementLevel, UnitInterval]]
    justifications: Mapping[str, str]


class Judge(ABC):
    def __init__(self, exp: Optional[Live], use_cache: bool = True):
        self.exp = exp
        self.use_cache = use_cache

    @abstractmethod
    def __enter__(self) -> "Judge":
        raise NotImplementedError

    @abstractmethod
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Any,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def judge_next_turn(
        self,
        resumes: Mapping[Any, str],
        interviews: Mapping[Any, Tuple[Mapping[str, str], ...]],
        rubrics: Mapping[Any, dict],
        max_workers: int
    ) -> Mapping[Any, Judgement]:
        raise NotImplementedError


class IndependentJudge(Judge):
    def __enter__(self) -> "IndependentJudge":
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")

        base_client = OpenAI(api_key=key)
        self.client = with_responses_cache(base_client) if self.use_cache else base_client
        
        self.model_name = "gpt-5-2025-08-07"
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def _system_prompt(
        self,
        rubric: dict,
        previous: Optional[Judgement] = None,
    ) -> str:
        # Default: no previous judgement injected.
        base = """
=== Context ===

You are an expert assessor. You will be given:
- A Rubric (evaluation dimensions and what “low / medium / high” look like for each)
- Evidence (resume + interview transcript so far)

(IMPORTANT) Before proceeding, you must internally construct a  Prior (your belief about the applicant’s 
true underlying level for each dimension BEFORE the latest applicant message.) If 

Key definition (important):
- Evidence Set (per dimension) is NOT a quantity or score.
  It is the current set of specific claims/observations we have about the applicant for that dimension,
  each of which may support low, medium, or high (or invalidate earlier claims).
  More items in the Evidence Set does not imply “higher”. Evidence can support any level.

=== Task ===

You must estimate the applicant’s true underlying level for each dimension as a Posterior distribution
over (low, medium, high) using ONLY the provided evidence (resume + transcript).

(IMPORTANT) You MUST start from a UNIFORM Prior for every dimension: P(low)=P(medium)=P(high)=1/3. Then, 
considering the ENTIRE Evidence Set so far, update that uniform prior into your Posterior.

Each applicant message can have exactly ONE of the following effects on the Evidence Set (at that point in 
time) of a given dimension:

1) No change:
    - Irrelevant, repeats or paraphrases what is already in the Evidence Set without adding new material information, or too vague to add/undo any specific claim.
    - In this case, keep the Prior unchanged.

2) Adds evidence:
    - The message adds a new, specific, relevant signal about the applicant for that dimension.
    - The added signal may support low OR medium OR high.
    - You should add probability mass to the one level that the new evidence best supports.
    - Just because the amount of evidence has increased does not mean that the likely level is "higher"
    - Be particularly careful to determine if "positive" signal best supports LOW or MEDIUM or HIGH.
    - Positive signal can still primarily support LOW if it does not satisfy the burden-of-proof for MEDIUM.

    Important warning: if the applicant indicates that they did not perform some action due to
    legitimate circumstances, then the inaction is NOT evidence of a lower level. Examples of
    such circumstances:
    - applicant had limited bandwidth
    - action was not necessary/appropriate
    - action was performed by another party
    - 

3) Subtracts (invalidates) evidence:
    - The message contradicts, retracts, corrects, or undermines existing evidence in the Evidence Set for that dimension.
    - Reduce probability mass on whatever level that invalidated evidence previously supported, and redistribute belief accordingly.
    - Evidence supporting any level (low, medium or high) can be invalidated.
    - Think of subtraction as removing or weakening a previously counted evidence item for that dimension.

    Common “subtraction” modes (examples):
    a) Direct retraction (removes a specific claim):
        - Earlier: “I led a team of 8 engineers.”  Latest: “Correction: I didn’t lead the team; I was an individual contributor.”
        → Remove/discount the leadership signal for that dimension.

    b) Contradiction / inconsistency (weakens earlier certainty):
        - Earlier: “I built the system end-to-end.”  Latest: “I only implemented a small component; another team owned the architecture.”
        → Reduce confidence in prior high-level ownership claims.

    c) Misspoke / oversimplified → later clarified (adjusts scope/precision, not necessarily “down”):
        - Earlier (compressed): “We migrated everything to Kubernetes.”
        Latest: “I oversimplified—my part was migrating two services; the platform team handled the cluster and the rest.”
        → Subtract the over-broad portion of the claim; keep the narrower, supported portion.

    d) Downgrading scope / responsibility (keeps involvement but removes ownership):
        - Earlier: “I owned the roadmap.”  Latest: “I contributed input, but my manager owned the roadmap.”
        → Subtract “ownership” evidence; keep “participation” evidence.

    e) Agency clarification that INVALIDATES prior ‘low’ evidence (boss made the call / applicant disagreed):
        - Earlier: “We shipped without tests because of deadlines.” (could be read as poor judgment/low quality bar)
        Latest: “That decision was made by my boss; I argued against it and proposed a phased test plan, but was overruled.”
        → Remove/discount evidence suggesting the applicant endorsed the low-standard decision; add/retain evidence of risk awareness/advocacy.
        (This is subtraction of evidence supporting LOW.)

    f) “Sounded like poor judgment” → later context removes the negative inference (debasing low):
        - Earlier: “I rolled back production by restarting servers.” (could imply ad-hoc ops)
        Latest: “To clarify: we used an automated rollback runbook; ‘restart’ was shorthand for reverting a deployment via our tooling.”
        → Subtract the earlier negative inference; keep the corrected, more credible version.

    g) Revealing a prior claim was mistaken or overstated (reduces strength/credibility of that item):
        - Earlier: “We reduced latency by 40%.”  Latest: “It was actually ~10%, and I’m not sure how it was measured.”
        → Discount the strength/credibility of the performance-impact evidence.

    h) Undermining credibility when pressed (only if meaningful, not mere brevity):
        - Earlier: “I designed the architecture.”  Latest (when asked): cannot explain key tradeoffs / components / constraints.
        → Treat the earlier strong claim as weaker; shift mass away from the level it supported.

    Note:
    - Subtraction is symmetric: you can invalidate evidence that previously supported low, medium, or high.
    - Do not punish normal uncertainty or humility; only subtract when the latest message meaningfully weakens or invalidates a specific prior claim or inference.
    - If the latest message is merely less detailed than earlier, but not contradictory or corrective, that is usually “No change,” not subtraction.

You must estimate 

=== Inputs ===

<Rubric>
{rubric}
</Rubric>

""".strip()

        return base.format(rubric=rubric)

    def _judge_one(
        self,
        interview_id: str,
        resume: str,
        interview: Tuple[Mapping[str, str], ...],
        rubric: dict,
        previous: Optional[Judgement] = None,
    ) -> tuple[str, Judgement]:
        rubric_keys = list(rubric.keys())
        rubric_schema = {
            "name": "posteriors",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    k: {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "levels": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "justification": {"type": "string"},
                        },
                        "required": ["levels", "justification"],
                    }
                    for k in rubric_keys
                },
                "required": rubric_keys,
            },
        }

        system_prompt = self._system_prompt(rubric=rubric, previous=previous)

        chat = [
            {
                "role": "assistant" if turn["role"] == "interviewer" else "user",
                "content": turn["message"],
            }
            for turn in interview
        ]

        payload: List[Mapping[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<Resume Start>{resume}<Resume End>"},
            *chat,
        ]

        #print(f"\n\nPAYLOAD START\n\n{payload}\n\nPAYLOAD END\n\n")

        response = self.client.responses.create(
            model=self.model_name,
            input=payload,
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": rubric_schema["name"],
                    "schema": rubric_schema["schema"],
                    "strict": True,
                }
            },
        )

        msg = next(x for x in response.output if x.type == "message")
        data = json.loads(msg.content[0].text)

        posteriors = {
            k: {
                "low": float(data[k]["levels"][0]),
                "medium": float(data[k]["levels"][1]),
                "high": float(data[k]["levels"][2]),
            }
            for k in rubric_keys
        }
        justifications = {k: str(data[k]["justification"]) for k in rubric_keys}

        return interview_id, Judgement(posteriors=posteriors, justifications=justifications)

    def judge_next_turn(
        self,
        resumes: Mapping[Any, str],
        interviews: Mapping[Any, Tuple[Mapping[str, str], ...]],
        rubrics: Mapping[Any, dict],
        max_workers: int = 32
    ) -> Mapping[Any, Judgement]:
        out: Dict[Any, Judgement] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self._judge_one, interview_id, resume, interviews[interview_id], rubrics[interview_id], None): interview_id
                for interview_id, resume in resumes.items()
            }
            for fut in as_completed(futures):
                interview_id, judgement = fut.result()
                out[interview_id] = judgement

        return out


class PreviousAwareJudge(IndependentJudge):
    def __enter__(self) -> "PreviousAwareJudge":
        super().__enter__()
        self._previous: Dict[str, Judgement] = {}
        return self

    def _system_prompt(self, rubric: dict, previous: Judgement) -> str:
        base = \
"""
=== Context ===

You are an expert assessor. You will be given:
- A Rubric (evaluation dimensions and what “low / medium / high” look like for each)
- A Prior (your belief about the applicant’s true underlying level for each dimension BEFORE the latest applicant message)
- Evidence (resume + interview transcript so far)

Key definition (important):
- Evidence Set (per dimension) is NOT a quantity or score.
  It is the current set of specific claims/observations we have about the applicant for that dimension,
  each of which may support low, medium, or high (or invalidate earlier claims).
  More items in the Evidence Set does not imply “higher”. Evidence can support any level.

=== Task ===

Update the Prior into a Posterior by considering ONLY how the latest applicant message changes the Evidence Set.
This is an incremental update: do not re-grade the entire history from scratch.

The latest applicant message can do exactly one of the following for a given dimension:

1) No change:
    - Irrelevant, repeats or paraphrases what is already in the Evidence Set without adding new material information, or too vague to add/undo any specific claim.
    - In this case, keep the Prior unchanged.

2) Adds evidence:
    - The message adds a new, specific, relevant signal about the applicant for that dimension.
    - The added signal may support low OR medium OR high.
    - You should add probability mass to the one level that the new evidence best supports.
    - Just because the amount of evidence has increased does not mean that the likely level is "higher"
    - Be particularly careful to determine if "positive" signal best supports LOW or MEDIUM or HIGH.
    - Positive signal can still primarily support LOW if it does not satisfy the burden-of-proof for MEDIUM.

    Important warning: if the applicant indicates that they did not perform some action due to
    legitimate circumstances, then the inaction is NOT evidence of a lower level. Examples of
    such circumstances:
    - applicant had limited bandwidth
    - action was not necessary/appropriate
    - action was performed by another party
    - 

3) Subtracts (invalidates) evidence:
    - The message contradicts, retracts, corrects, or undermines existing evidence in the Evidence Set for that dimension.
    - Reduce probability mass on whatever level that invalidated evidence previously supported, and redistribute belief accordingly.
    - Evidence supporting any level (low, medium or high) can be invalidated.
    - Think of subtraction as removing or weakening a previously counted evidence item for that dimension.

    Common “subtraction” modes (examples):
    a) Direct retraction (removes a specific claim):
        - Earlier: “I led a team of 8 engineers.”  Latest: “Correction: I didn’t lead the team; I was an individual contributor.”
        → Remove/discount the leadership signal for that dimension.

    b) Contradiction / inconsistency (weakens earlier certainty):
        - Earlier: “I built the system end-to-end.”  Latest: “I only implemented a small component; another team owned the architecture.”
        → Reduce confidence in prior high-level ownership claims.

    c) Misspoke / oversimplified → later clarified (adjusts scope/precision, not necessarily “down”):
        - Earlier (compressed): “We migrated everything to Kubernetes.”
        Latest: “I oversimplified—my part was migrating two services; the platform team handled the cluster and the rest.”
        → Subtract the over-broad portion of the claim; keep the narrower, supported portion.

    d) Downgrading scope / responsibility (keeps involvement but removes ownership):
        - Earlier: “I owned the roadmap.”  Latest: “I contributed input, but my manager owned the roadmap.”
        → Subtract “ownership” evidence; keep “participation” evidence.

    e) Agency clarification that INVALIDATES prior ‘low’ evidence (boss made the call / applicant disagreed):
        - Earlier: “We shipped without tests because of deadlines.” (could be read as poor judgment/low quality bar)
        Latest: “That decision was made by my boss; I argued against it and proposed a phased test plan, but was overruled.”
        → Remove/discount evidence suggesting the applicant endorsed the low-standard decision; add/retain evidence of risk awareness/advocacy.
        (This is subtraction of evidence supporting LOW.)

    f) “Sounded like poor judgment” → later context removes the negative inference (debasing low):
        - Earlier: “I rolled back production by restarting servers.” (could imply ad-hoc ops)
        Latest: “To clarify: we used an automated rollback runbook; ‘restart’ was shorthand for reverting a deployment via our tooling.”
        → Subtract the earlier negative inference; keep the corrected, more credible version.

    g) Revealing a prior claim was mistaken or overstated (reduces strength/credibility of that item):
        - Earlier: “We reduced latency by 40%.”  Latest: “It was actually ~10%, and I’m not sure how it was measured.”
        → Discount the strength/credibility of the performance-impact evidence.

    h) Undermining credibility when pressed (only if meaningful, not mere brevity):
        - Earlier: “I designed the architecture.”  Latest (when asked): cannot explain key tradeoffs / components / constraints.
        → Treat the earlier strong claim as weaker; shift mass away from the level it supported.

    Note:
    - Subtraction is symmetric: you can invalidate evidence that previously supported low, medium, or high.
    - Do not punish normal uncertainty or humility; only subtract when the latest message meaningfully weakens or invalidates a specific prior claim or inference.
    - If the latest message is merely less detailed than earlier, but not contradictory or corrective, that is usually “No change,” not subtraction.


=== Inputs ===

<Rubric>
{rubric}
</Rubric>

<Prior>
{prior}
</Prior>
"""

        prior_blob = {
            dim: {
                "posteriors": {
                    "low": float(p["low"]),
                    "medium": float(p["medium"]),
                    "high": float(p["high"]),
                },
                "justification": str(previous.justifications.get(dim, "")),
            }
            for dim, p in previous.posteriors.items()
        }

        return base.strip().format(
            rubric=rubric,
            prior=json.dumps(prior_blob, ensure_ascii=False),
        )

    def _uniform_prior(self, rubric: dict) -> Judgement:
        dims = list(rubric.keys())
        posteriors = {d: {"low": 1/3, "medium": 1/3, "high": 1/3} for d in dims}
        justifications = {d: "" for d in dims}
        return Judgement(posteriors=posteriors, justifications=justifications)

    def judge_next_turn(
        self,
        resumes: Mapping[Any, str],
        interviews: Mapping[Any, Tuple[Mapping[str, str], ...]],
        rubrics: Mapping[Any, dict],
        max_workers: int = 32
    ) -> Mapping[Any, Judgement]:
        out: Dict[Any, Judgement] = {}

        prev_snapshot = dict(self._previous)
        priors: Dict[str, Judgement] = {
            interview_id: (prev_snapshot.get(interview_id) or self._uniform_prior(rubrics[interview_id]))
            for interview_id in resumes
        }

        max_workers = min(32, max(1, len(resumes)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    self._judge_one,
                    interview_id,
                    resume,
                    interviews[interview_id],
                    rubrics[interview_id],
                    priors[interview_id],
                ): interview_id
                for interview_id, resume in resumes.items()
            }

            for fut in as_completed(futures):
                interview_id, judgement = fut.result()
                out[interview_id] = judgement

        self._previous.update(out)
        return out


JUDGES: Mapping[JudgeName, type[Judge]] = {
    "independent": IndependentJudge,
    "previous-judgement-aware": PreviousAwareJudge,
}
