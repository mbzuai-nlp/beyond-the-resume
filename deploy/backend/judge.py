import json
from typing import Mapping, Sequence

from openai import AsyncOpenAI

from interviewer import MODEL_NAME


SYSTEM_PROMPT = """
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
""".strip()

RESUME_SYSTEM_PROMPT = """
=== Context ===

You are an expert assessor. You will be given:
- A Rubric (evaluation dimensions and what “low / medium / high” look like for each)
- A Resume

=== Task ===

Estimate the applicant's true underlying level for each dimension as a Posterior distribution over
(low, medium, high) using ONLY the resume.

Start from a UNIFORM Prior for every dimension: P(low)=P(medium)=P(high)=1/3.
Update that prior only where the resume provides specific evidence. Absence of evidence is not evidence
of a low level.

=== Inputs ===

<Rubric>
{rubric}
</Rubric>
""".strip()


def uniform_prior(rubric: Mapping[str, object]) -> dict:
    return {
        dimension: {
            "posteriors": {
                "low": 1 / 3,
                "medium": 1 / 3,
                "high": 1 / 3,
            },
            "justification": "",
        }
        for dimension in rubric
    }


def belief_format(dimensions: Sequence[str], name: str) -> dict:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            dimension: {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "low": {"type": "number", "minimum": 0, "maximum": 1},
                    "medium": {"type": "number", "minimum": 0, "maximum": 1},
                    "high": {"type": "number", "minimum": 0, "maximum": 1},
                    "justification": {"type": "string"},
                },
                "required": ["low", "medium", "high", "justification"],
            }
            for dimension in dimensions
        },
        "required": list(dimensions),
    }
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }
    }


def parse_belief(response_text: str, dimensions: Sequence[str]) -> tuple[dict, dict]:
    data = json.loads(response_text)
    posteriors = {
        dimension: {
            level: float(data[dimension][level])
            for level in ("low", "medium", "high")
        }
        for dimension in dimensions
    }
    justifications = {
        dimension: str(data[dimension]["justification"])
        for dimension in dimensions
    }
    return posteriors, justifications


async def get_resume_belief(
    client: AsyncOpenAI,
    resume: str,
    rubric: Mapping[str, object],
) -> tuple[dict, dict]:
    dimensions = list(rubric)
    response = await client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": RESUME_SYSTEM_PROMPT.format(
                    rubric=json.dumps(rubric, ensure_ascii=False)
                ),
            },
            {"role": "user", "content": f"<Resume Start>{resume}<Resume End>"},
        ],
        reasoning={"effort": "low"},
        text=belief_format(dimensions, "resume_belief"),
    )
    return parse_belief(response.output_text, dimensions)


async def get_belief_update(
    client: AsyncOpenAI,
    resume: str,
    interview: Sequence[Mapping[str, str]],
    rubric: Mapping[str, object],
    previous: Mapping[str, object] | None,
) -> tuple[dict, dict]:
    dimensions = list(rubric)
    prior = previous or uniform_prior(rubric)
    system_prompt = SYSTEM_PROMPT.format(
        rubric=json.dumps(rubric, ensure_ascii=False),
        prior=json.dumps(prior, ensure_ascii=False),
    )
    chat = [
        {
            "role": "assistant" if turn["role"] == "interviewer" else "user",
            "content": turn["message"],
        }
        for turn in interview
    ]
    response = await client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<Resume Start>{resume}<Resume End>"},
            *chat,
        ],
        reasoning={"effort": "low"},
        text=belief_format(dimensions, "belief_update"),
    )
    return parse_belief(response.output_text, dimensions)
