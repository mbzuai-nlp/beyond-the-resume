from src.judge import JudgeName, JUDGES, Judgement
from src.llm_response_cache import with_responses_cache

from dataclasses import dataclass
from dvclive import Live
from pathlib import Path
from typing import Mapping, Dict, Tuple, List
from openai import OpenAI
import json
import polars as pl
from copy import deepcopy
import random
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


load_dotenv()


@dataclass(frozen=True)
class Config:
    interviewer_name: str
    judge_name: JudgeName
    data_dir: Path
    save_path: Path
    num_interview_turns: int = 12


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CLIENT = with_responses_cache(OpenAI(api_key=OPENAI_API_KEY))
OPENAI_MODEL_NAME = "gpt-5-2025-08-07"

MAX_WORKERS = 32

PERSONALITY_TYPES = (
    "timid",
    "jovial",
    "cold",
    "frank",
    "long-whinded",
    "relaxed",
    "combative",
    "friendly",
)

LATENT_LEVELS = ("low", "medium", "high")


def _load_profiles(
    data_dir: Path,
) -> Mapping[Tuple[str, str], Tuple[str, Dict]]:
    """Loads mapping of ((resume_id, archetype_id), (resume, archetype))"""
    resumes = {}
    resumes_dir = data_dir / "resumes"
    for json_path in resumes_dir.glob("*.json"):
        resume_id = json_path.stem
        with open(json_path, "r") as f:
            resume = json.load(f)["text"]
        resumes[resume_id] = resume

    archetypes_path = data_dir / "applicant_archetypes.json"
    with open(archetypes_path, "r") as f:
        domain_archetypes = json.load(f)

    out = {}
    for rid, resume in resumes.items():
        domain = rid.split("_")[0]

        for aid, archetype in domain_archetypes[domain].items():
            out[(rid, aid)] = (resume, archetype)

    return out


def _load_rubrics(data_dir: Path, profile_ids: Tuple[Tuple[str, str], ...]) -> Mapping[Tuple[str, str], dict]:
    rubrics_dir = data_dir / "rubrics"
    rubrics: Dict[str, dict] = {}
    for json_path in rubrics_dir.glob("*.json"):
        rubric_id = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            rubrics[rubric_id] = json.load(f)

    resume_rubrics: Dict[str, dict] = {}
    for rid, aid in profile_ids:
        domain = rid.split("_")[0]
        resume_rubrics[(rid, aid)] = deepcopy(rubrics[domain])
    return resume_rubrics


def _get_applicant_messages(
    resumes: Mapping[Tuple[str, str], str],
    interviews: Mapping[Tuple[str, str], List[Mapping[str, str]]],
    personalities: Mapping[Tuple[str, str], str],
    rubrics: Mapping[str, dict],
    archetypes: Mapping[Tuple[str, str], dict],
) -> Mapping[Tuple[str, str], str]:
    def refine_applicant_response(
        proposed_response: str,
        rubric: Dict,
        archetype: dict,
    ) -> str:
        """
        Refinement step: push the proposed response DOWN into the archetype by removing
        implications of higher levels. Intentionally does NOT include the interviewer question.
        """
        system_prompt = """
=== Task ===
You are a "response downgrader" for synthetic interview data.

Given:
- a proposed applicant response
- the rubric
- the applicant archetype (target levels per dimension)

Rewrite the response so it is faithful to the archetype by REMOVING any implication of a
higher level.

=== How to Downgrade ===

(Important) Firstly make sure to downgrade only along the relevant dimensions.

When these instructions refer to the Archetype, they specifically refer to the subset of dimensions
relevant to the response.

If the relevant Archetype dimension is LOW:
1. You must remove technical terms, concepts and tooling that would likely be used by an applicant 
   of a MEDIUM.
2. Identify information that is arguably consistent with a LOW-MEDIUM level and remove it. This 
   means that even if there is only a tiny chance that such detail could be misinterpreted as a MEDIUM
   level you must remove it. You may use the following techniques to accomplish this:
   - delete details,
   - replace specifics with vague language,
   - deflect ownership/attribution to other teams/processes/systems (e.g. "X was handled by another team"),
   - or admit uncertainty / not remembering.
3. Curb rigour and methodology that would likely be exhibited by an applicant of a MEDIUM/HIGH level.

If the relevant Archetype dimension is MEDIUM:
1. You must remove technical terms, concepts and tooling that would likely be used by an applicant 
   of a HIGH.
2. Identify information that is arguably consistent with a MEDIUM-HIGH level and remove it. This 
   means that even if there is only a tiny chance that such detail could be misinterpreted as a HIGH
   level you must remove it. You may use the following techniques to accomplish this:
   - delete details,
   - replace specifics with vague language,
   - deflect ownership/attribution to other teams/processes/systems (e.g. "X was handled by another team"),
   - or admit uncertainty / not remembering.
3. Curb rigour and methodology that would likely be exhibited by an applicant of a HIGH level.

If the relevant Archetype dimension is HIGH:
1. Reduce specificity so that the response reads like conversational English and less like an enumeration
   HIGH level cues. Do not downgrade the semantics or implied competency, just prevent it from reading
   like a system log.

For ALL levels:
(STRICT) You must not cheat with negative self-evaluation. Examples of negative self-evaluations:
- "I struggled with X"
- "My organisation was bad"
- "I didn't keep good records"
- "I only looked at X and didn't do Y"
- "We did X rather than Y"

=== Format and Style ===

- Output MUST be 1-2 sentences.
- Write the response in the style expected of the level relevant:
    - LOW: Presents incompetence, deflects responsibility and generally struggles to explain demonstrating
           limited understanding
    - MEDIUM: Presents some competence but not excellence and has clear gaps
    - HIGH: Presents as an authority, uses specifics, does not omit key information
- (STRICT) You must never invent new detail! You are only permitted to downgrade, never to upgrade!

=== Inputs ===
<Archetype>
{archetype}
</Archetype>

<Rubric>
{rubric}
</Rubric>
""".strip()

        payload = [
            {
                "role": "system",
                "content": system_prompt.format(
                    archetype=json.dumps(archetype, ensure_ascii=False, indent=2),
                    rubric=json.dumps(rubric, ensure_ascii=False, indent=2),
                ),
            },
            {
                "role": "user",
                "content": f"<ProposedResponse>\n{proposed_response}\n</ProposedResponse>",
            },
        ]

        resp = OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL_NAME,
            input=payload,
            reasoning={"effort": "low"},
        )
        refined = resp.output_text
        return refined

    def get_applicant_message(
        profile_id: Tuple[str, str],
        resume: str,
        interview: List[Mapping[str, str]],
        personality: str,
        rubric: Dict,
        archetype: dict,
    ) -> tuple[Tuple[str, str], str]:
        system_prompt_template = """
=== Task ===

You are helping out with creating synthetic training data for an AI interview system
by playing the role of Applicant. You are tasked with conversing with an Interviewer
in a way that is faithful to the provided Archetype.

The Archetype defines a subspace within the space of knowledge, skills and dispositions
(KSD) spanned by the rubric. We consider this subspace to define your specific KSD.

Knowledge refers to what you understand. Skills refer to the things you can do. 
Dispositions refer to your behaviours and tendencies.

=== Global Response Constraints ===

G1. (STRICT) You must not leak your task to the Interviewer and must not cheat by
    tagging your response with the rubric level.
G2. Each response must be 1-2 sentences.
G3. Each response must be written like someone who exhibits the KSD defined by your archetype.
G4. Your response must be clearly anchored to at least 1 explicit KSD of the corresponding
    Rubric level.
G5. Never volunteer extra information that was not directly asked about by the interviewer.

=== Self-Preservation Constraint (STRICT) ===

SP1. Never use negative self-evaluation. Examples of negative self-evaluations:
     - "I struggled with X"
     - "My organisation was bad"
     - "I didn't keep good records"
SP2. Show, don't confess. Express gaps, shortcomings and inability indirectly
     via concrete behaviors rather than negative judgements.

=== Using the Resume ===

R1. (STRICT) The provided Resume does must NOT affect your KSD levels implied by your 
    response. Only the Archetype is allowed to govern your levels.

R2. You must ONLY use the Resume for context about your experiences.

=== Show Don't Tell (VERY STRICT) ===

S1. You must NOT explicitly list the expected behaviours and skills of your Archetype's KSD.
   You must IMPLY your KSD via example, sharing opinion and discussing experiences. If
   a judge reads your reponse and determines that you have just listed terms for the sake of
   matching the rubric then you have failed. You must truly embody your archetype and pretend
   you do not know what the rubric is but instead naturaly inhabit the Archetype and thus
   communicate your KSD through implications. 

S2. (STRICT) If asked about a specific experience, you MUST contextualise your answer. To
   contextualise your answer means to relate any facts and outcomes to the experience. You
   may invent specific details and examples aligned with the KSD of your archetype to contextualise
   your response. You must not invent detail that inflates your experience, you must make sure it
   is wholly congruent with the KSD of your archetype.

=== No Volunteering (SUPER STRICT) ===

NV1. You MUST NEVER volunteer information that is not directly asked about by the interviewer. Do
     NOT volunteer extra detail, methodology, justification, behaviour or outcomes unless directly
     asked by the interviewer.
NV2. Answer ONLY what is asked of you. Remember, you are helping create synthetic training data for
     training an interviewer model. This obviously means that we want to be able to discern between
     good and bad interviewer questions and can only do this if you ONLY address what is asked. If you
     volunteer extra information without being asked directly for it then we will be unable to tell
     the difference between a good and bad question because you're providing the same information 
     regardless. Only answer what is asked of you, or else you have failed.
NV3. Therefore, a vague or open-ended question should be answered vaguely because you are FORBIDDEN
     from volunteering information not directly asked of you and by the anture of open-ended questions,
     they do not probe for specifics.

=== Inputs ===

Personality: {personality}

<Archetype>
{archetype}
</Archetype>

<Rubric>
{rubric}
</Rubric>

<Resume>
{resume}
</Resume>
""".strip()

        system_prompt = system_prompt_template.format(
            personality=personality,
            resume=resume,
            archetype=json.dumps(archetype, ensure_ascii=False, indent=2),
            rubric=json.dumps(rubric, ensure_ascii=False, indent=2),
        )

        chat = [
            {
                "role": "assistant" if turn["role"] == "applicant" else "user",
                "content": turn["message"],
            }
            for turn in interview
        ]

        payload = [{"role": "system", "content": system_prompt}, *chat]

        response = OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL_NAME,
            input=payload,
            reasoning={"effort": "low"},
        )

        proposed = response.output_text

        refined = refine_applicant_response(
            proposed_response=proposed,
            rubric=rubric,
            archetype=archetype,
        )

        return profile_id, refined

    out: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                get_applicant_message,
                pid,
                resume,
                interviews[pid],
                personalities[pid],
                rubrics[pid],
                archetypes[pid],
            ): pid
            for pid, resume in resumes.items()
        }
        for fut in as_completed(futures):
            pid, message = fut.result()
            out[pid] = message
    return out


def _write_simulation_parquet(
    save_path: Path,
    resumes: Mapping[Tuple[str, str], str],
    rubrics: Mapping[Tuple[str, str], dict],
    interviews: Mapping[Tuple[str, str], List[Mapping[str, str]]],
    personalities: Mapping[Tuple[str, str], str],
    archetypes: Mapping[Tuple[str, str], dict],
    judgement_histories: Mapping[Tuple[str, str], List[Judgement]],
) -> None:
    rows: List[dict] = []
    for (resume_id, archetype_id), resume_text in resumes.items():
        pid = (resume_id, archetype_id)
        turns = interviews[pid]
        jhist = judgement_histories[pid]

        rows.append(
            {
                "resume_id": resume_id,
                "archetype_id": archetype_id,
                "resume": resume_text,
                "rubric_json": json.dumps(rubrics[pid], ensure_ascii=False, indent=2),
                "personality": personalities[pid],
                "archetype_json": json.dumps(archetypes[pid], ensure_ascii=False, indent=2),
                "interview_turns_json": json.dumps(list(turns), ensure_ascii=False, indent=2),
                "judgement_history_posteriors_json": json.dumps([j.posteriors for j in jhist], ensure_ascii=False, indent=2),
                "judgement_history_justifications_json": json.dumps([j.justifications for j in jhist], ensure_ascii=False, indent=2),
            }
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(save_path)


def get_interviewer_messages(
    config: Config,
    resumes: Mapping[Tuple[str, str], str],
    interviews: Mapping[Tuple[str, str], List[Mapping[str, str]]],
    rubrics: Mapping[Tuple[str, str], dict],
    judgements: Mapping[Tuple[str, str], Judgement]
) -> Mapping[Tuple[str, str], str]:
    def get_interviewer_message(
        profile_id: Tuple[str, str],
        resume: str,
        interview: List[Mapping[str, str]],
        rubric: dict,
        judgement: Judgement,
    ) -> tuple[Tuple[str, str], str]:
        interviewer = config.interviewer_name
        print(f"USING INTERVIEWER: {interviewer}")

        if interviewer == "latest-judgement-aware":
            system_prompt_template = \
"""
=== Task ===
You are an interviewer asking the applicant about their resume and experiences based on the provided rubric.

You will be provided with the CurrentJudgement which represents a judge's belief of the applicant's
levels over the rubric dimensions. Your task is to elicit information from the applicant to stress-test the
hypotheses presented in the current judgement. This looks like:

- Eliciting information to disambiguate between competing levels
- Eliciting information to verify if a current hypothesis is true

Remember your overall goal is to elicit information such that the judge gets the truest possible picture of the
applicant.

=== Information Gathering ===

- Do not probe about exact numbers and specifics that the applicant likely would not be at liberty to discuss
- (STRICT) Over the course of conversation you must make sure you span the resume and
the dimensions of the rubric
- Ask follow up questions when you think there is more pertinent information to be elicited
- (STRICT) Stop pursing a conversational thread if it becomes apparent that further probing is unlikely to introduce new
evidence that would change the belief of the Applicant's level
- Avoid spoonfeeding the applicant - this means do NOT ask a questions and suggest possible answers or explanations or approaches
- (STRICT) When asking about a NEW topic, it must be grounded in a Resume item and NOT some generic question

=== No Presuppositioning (STRICT) ===

- Do NOT assume an outcome was achieved or a behavior occurred.
- Do NOT embed conclusions in the question (e.g., "so the site stayed usable...", "to ensure X").

=== Style Enforcement ===

- Always acknowledge the applicant's last message
- Avoid all formatting, markdown and paragraphs etc. you should have no newline chars
- The conversation should be fluid. Do not sound like a robot reading off a checklist
- This is not an exam, this is a casual conversation
- You must only ask for one coherent piece of information at a time
- The interview must feel like a casual conversation so be friendly and ensure your messages flow organically

If your response is laborious to read or does not read like a quick question one would ask during a normal conversation then you have failed.

=== Inputs ===

Budget: {num_turns} interviewer turns

<Rubric>
{rubric}
</Rubric>

<Resume>
{resume}
</Resume>

<Judgement>
{judgement}
</Judgement>
"""
            system_prompt = system_prompt_template.format(
                resume=resume,
                rubric=json.dumps(rubric, ensure_ascii=False, indent=2),
                judgement=json.dumps(judgement.posteriors, ensure_ascii=False, indent=2),
                num_turns=config.num_interview_turns
            )
        elif interviewer == "judgement-unaware":
            system_prompt_template = \
"""
=== Task ===
You are an interviewer asking the applicant about their resume and experiences based on the provided rubric.

=== Information Gathering ===

- Do not probe about exact numbers and specifics that the applicant likely would not be at liberty to discuss
- (STRICT) Over the course of conversation you must make sure you span the resume and
the dimensions of the rubric
- Ask follow up questions when you think there is more pertinent information to be elicited
- (STRICT) Stop pursing a conversational thread if it becomes apparent that further probing is unlikely to introduce new
evidence that would change the belief of the Applicant's level
- Avoid spoonfeeding the applicant - this means do NOT ask a questions and suggest possible answers or explanations or approaches
- (STRICT) When asking about a NEW topic, it must be grounded in a Resume item and NOT some generic question

=== No Presuppositioning (STRICT) ===

- Do NOT assume an outcome was achieved or a behavior occurred.
- Do NOT embed conclusions in the question (e.g., "so the site stayed usable...", "to ensure X").

=== Style Enforcement ===

- Always acknowledge the applicant's last message
- Avoid all formatting, markdown and paragraphs etc. you should have no newline chars
- The conversation should be fluid. Do not sound like a robot reading off a checklist
- This is not an exam, this is a casual conversation
- You must only ask for one coherent piece of information at a time
- The interview must feel like a casual conversation so be friendly and ensure your messages flow organically

If your response is laborious to read or does not read like a quick question one would ask during a normal conversation then you have failed.

=== Inputs ===

Budget: {num_turns} interviewer turns

<Rubric>
{rubric}
</Rubric>

<Resume>
{resume}
</Resume>
"""
            system_prompt = system_prompt_template.format(
                resume=resume,
                rubric=json.dumps(rubric, ensure_ascii=False, indent=2),
                num_turns=config.num_interview_turns
            )
        elif interviewer == "rubric-unaware":
            system_prompt_template = \
"""
=== Task ===
You are an interviewer asking the applicant about their resume and experiences.

=== Information Gathering ===

- Do not probe about exact numbers and specifics that the applicant likely would not be at liberty to discuss
- (STRICT) Over the course of conversation you must make sure you span the resume.
- Ask follow up questions when you think there is more pertinent information to be elicited
- (STRICT) Stop pursing a conversational thread if it becomes apparent that further probing is unlikely to introduce new
evidence that would change the belief of the Applicant's level
- Avoid spoonfeeding the applicant - this means do NOT ask a questions and suggest possible answers or explanations or approaches
- (STRICT) When asking about a NEW topic, it must be grounded in a Resume item and NOT some generic question

=== No Presuppositioning (STRICT) ===

- Do NOT assume an outcome was achieved or a behavior occurred.
- Do NOT embed conclusions in the question (e.g., "so the site stayed usable...", "to ensure X").

=== Style Enforcement ===

- Always acknowledge the applicant's last message
- Avoid all formatting, markdown and paragraphs etc. you should have no newline chars
- The conversation should be fluid. Do not sound like a robot reading off a checklist
- This is not an exam, this is a casual conversation
- You must only ask for one coherent piece of information at a time
- The interview must feel like a casual conversation so be friendly and ensure your messages flow organically

If your response is laborious to read or does not read like a quick question one would ask during a normal conversation then you have failed.

=== Inputs ===

Budget: {num_turns} interviewer turns

<Resume>
{resume}
</Resume>
"""
            system_prompt = system_prompt_template.format(
                resume=resume,
                num_turns=config.num_interview_turns
            )
        elif interviewer == "shallow-resume-screen":
            system_prompt_template = \
"""
=== Task ===
You are conducting a brief, friendly initial screening interview about the applicant's resume.

Your job is NOT to deeply assess technical ability. Your job is to keep the conversation moving and get a light overview of their experiences.

=== Interview Style (INTENTIONALLY SHALLOW) ===

- Be warm and conversational
- Ask simple, broad, resume-grounded questions
- Focus on what they worked on, what the team did, what the project was about, and what they enjoyed
- Prefer breadth over depth (move on quickly rather than drilling in)
- Avoid deep follow-ups about methodology, trade-offs, debugging, validation, failure analysis, or decision criteria
- Avoid trying to infer or test skill level directly
- Avoid adversarial probing or stress-testing claims
- Do not ask the applicant to justify why they chose a specific approach
- Do not ask multi-part questions
- Do not spoon-feed possible answers
- (STRICT) When asking about a NEW topic, it must be grounded in a Resume item
- If the applicant gives a detailed answer, acknowledge it briefly and ask a new simple question

=== No Presuppositioning (STRICT) ===

- Do NOT assume an outcome was achieved or a behavior occurred
- Do NOT embed conclusions in the question

=== Style Enforcement ===

- Always acknowledge the applicant's last message
- Avoid formatting, markdown, bullet points, and newlines
- Sound like a normal quick recruiter screen, not a technical interviewer
- Ask only one coherent question at a time
- Keep questions easy to answer and conversational

If your question sounds like it is trying to measure competence precisely, you have failed.

=== Inputs ===

Budget: {num_turns} interviewer turns

<Resume>
{resume}
</Resume>
"""
            system_prompt = system_prompt_template.format(
                resume=resume,
                num_turns=config.num_interview_turns
            )
        else:
            raise NotImplementedError()

        chat = [
            {
                "role": "assistant" if turn["role"] == "interviewer" else "user",
                "content": turn["message"],
            }
            for turn in interview
        ]

        payload = [{"role": "system", "content": system_prompt}, *chat]

        response = OPENAI_CLIENT.responses.create(
            model=OPENAI_MODEL_NAME,
            input=payload,
            reasoning={"effort": "low"},
        )

        return profile_id, response.output_text

    out: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(
                get_interviewer_message,
                pid,
                resume,
                interviews[pid],
                rubrics[pid],
                judgements[pid],
            ): pid
            for pid, resume in resumes.items()
        }
        for fut in as_completed(futures):
            pid, message = fut.result()
            out[pid] = message
    return out


def run_interview_simulations(
    config: Config,
    exp: Live,
    **judge_kwargs,
):
    judge = JUDGES[config.judge_name]

    profiles = _load_profiles(config.data_dir)
    profile_ids = sorted(list(profiles.keys()))

    # random sample
    if False:
        random.shuffle(profile_ids)
        profile_ids = profile_ids[:30]
    
    resumes = {pid: l[0] for pid, l in profiles.items() if pid in profile_ids}
    archetypes = {pid: l[1] for pid, l in profiles.items() if pid in profile_ids}
    
    rubrics = _load_rubrics(config.data_dir, profile_ids)

    applicant_personalities = {pid: random.choice(PERSONALITY_TYPES) for pid in profile_ids}

    judgement_histories: Dict[Tuple[str, str], List[Judgement]] = {pid: [] for pid in profile_ids}

    with judge(exp=exp, **judge_kwargs) as j:
        interviews: Dict[Tuple[str, str], List[Mapping[str, str]]] = {pid: [] for pid in profile_ids}

        # Resume only
        judgements = j.judge_next_turn(
            resumes=resumes,
            interviews=interviews,
            rubrics=rubrics,
            max_workers=MAX_WORKERS,
        )
        for rid, judgement in judgements.items():
            judgement_histories[rid].append(judgement)

        for _t in range(config.num_interview_turns):
            print(f"PROCESSING TURN {_t}")

            interviewer_messages = get_interviewer_messages(
                config=config,
                resumes=resumes,
                interviews=interviews,
                rubrics=rubrics,
                judgements=judgements
            )

            for pid, interviewer_message in interviewer_messages.items():
                interviews[pid].append({"role": "interviewer", "message": interviewer_message})

            applicant_messages = _get_applicant_messages(
                resumes=resumes,
                interviews=interviews,
                personalities=applicant_personalities,
                rubrics=rubrics,
                archetypes=archetypes,
            )

            for pid, applicant_message in applicant_messages.items():
                interviews[pid].append({"role": "applicant", "message": applicant_message})

            # After next turn
            judgements = j.judge_next_turn(
                resumes=resumes,
                interviews=interviews,
                rubrics=rubrics,
                max_workers=MAX_WORKERS,
            )
            for rid, judgement in judgements.items():
                judgement_histories[rid].append(judgement)

    _write_simulation_parquet(
        save_path=config.save_path,
        resumes=resumes,
        rubrics=rubrics,
        interviews=interviews,
        personalities=applicant_personalities,
        archetypes=archetypes,
        judgement_histories=judgement_histories,
    )
