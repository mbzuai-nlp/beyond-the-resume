from src.judge import Judgement
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from typing import Tuple, Mapping

load_dotenv()

OPENAI_AI_KEY = os.getenv("OPENAI_AI_KEY")
CLIENT = AsyncOpenAI(api_key=OPENAI_AI_KEY)

MODEL_NAME = "gpt-5-2025-08-07"


async def upload_resume_file(resume_file_path: str) -> str:
    with open(resume_file_path, "rb") as f:
        file_obj = await CLIENT.files.create(
            file=f,
            purpose="assistants"
        )
    return file_obj.id


async def get_interviewer_message(
    resume: str,
    interview: Tuple[Mapping[str, str], ...],
    rubric: Mapping[str, str],
    max_turns: int
) -> str:
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
        rubric=rubric,
        resume=resume,
        num_turns=max_turns
    )

    chat = [
        {
            "role": "assistant" if turn["role"] == "interviewer" else "user",
            "content": turn["message"],
        }
        for turn in interview
    ]

    payload = [
        {"role": "system", "content": system_prompt},
        *chat,
    ]

    response = await CLIENT.responses.create(
        model=MODEL_NAME,
        input=payload,
        reasoning={"effort": "low"},
    )

    return response.output_text


async def parse_resume(resume_file_id: str) -> str:
    """
    Extracts and returns the resume as plain text from an uploaded file_id.
    """
    response = await CLIENT.responses.create(
        model=MODEL_NAME,
        reasoning={"effort": "minimal"},
        input=[
            {
                "role": "system",
                "content": (
                    "You extract resume content. Return ONLY the resume text as plain UTF-8 text. "
                    "No markdown, no bullet reformatting, no commentary, no JSON—just the text."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the full resume text from this file."},
                    {"type": "input_file", "file_id": resume_file_id},
                ],
            },
        ],
    )
    return response.output_text.strip()