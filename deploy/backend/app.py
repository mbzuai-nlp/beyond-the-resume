import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
import logging
import os

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import Response
from openai import AsyncOpenAI
from pydantic import BaseModel

from database import (
    add_belief_update,
    add_message,
    create_application,
    get_application,
    get_interview,
    get_judge_context,
    get_resume,
    get_resume_text,
    healthcheck,
    initialize_database,
    list_applications,
)
from interviewer import get_interviewer_message, parse_resume, upload_resume_file
from judge import get_belief_update, get_resume_belief
from rubric import load_rubric


async def run_judge_worker(app):
    while True:
        kind, email, applicant_message_id = await app.state.judge_queue.get()
        try:
            rubric = {
                dimension["id"]: {
                    "name": dimension["name"],
                    "levels": dimension["levels"],
                }
                for dimension in load_rubric()["dimensions"]
            }
            if kind == "resume":
                posteriors, justifications = await get_resume_belief(
                    app.state.openai,
                    get_resume_text(email),
                    rubric,
                )
            else:
                resume_text, interview, previous = get_judge_context(
                    email,
                    applicant_message_id,
                )
                posteriors, justifications = await get_belief_update(
                    app.state.openai,
                    resume_text,
                    interview,
                    rubric,
                    previous,
                )
            add_belief_update(
                email,
                applicant_message_id,
                posteriors,
                justifications,
            )
        except Exception:
            logging.exception("Belief update failed for %s", email)
        finally:
            app.state.judge_queue.task_done()


@asynccontextmanager
async def lifespan(app):
    initialize_database()
    app.state.openai = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    app.state.judge_queue = asyncio.Queue()
    app.state.judge_worker = asyncio.create_task(run_judge_worker(app))
    yield
    app.state.judge_worker.cancel()
    with suppress(asyncio.CancelledError):
        await app.state.judge_worker
    await app.state.openai.close()


app = FastAPI(lifespan=lifespan)


class ApplicantMessage(BaseModel):
    email: str
    content: str


async def create_interviewer_message(client, email):
    resume_text, interview = get_interview(email)
    generation_started_at = datetime.now(timezone.utc)
    content = await get_interviewer_message(
        client,
        resume_text,
        interview,
        load_rubric(),
        int(os.environ["INTERVIEW_MAX_TURNS"]),
    )
    generation_completed_at = datetime.now(timezone.utc)
    return add_message(
        email,
        "interviewer",
        content,
        generation_started_at=generation_started_at,
        generation_completed_at=generation_completed_at,
    )


@app.get("/api/health")
def health():
    healthcheck()
    return {"status": "ok", "database": "connected"}


@app.post("/api/applications")
async def start_application(
    request: Request,
    email: str = Form(),
    resume: UploadFile = File(),
):
    resume_data = await resume.read()
    resume_file_id = await upload_resume_file(
        request.app.state.openai,
        resume.filename,
        resume_data,
    )
    resume_text = await parse_resume(request.app.state.openai, resume_file_id)
    application = create_application(
        email=email,
        resume_filename=resume.filename,
        resume_content_type=resume.content_type or "application/octet-stream",
        resume_data=resume_data,
        resume_text=resume_text,
    )
    interviewer_message = await create_interviewer_message(
        request.app.state.openai,
        email,
    )
    request.app.state.judge_queue.put_nowait(("resume", email, None))
    return {
        "application": application,
        "interviewer_message": interviewer_message,
        "interview_complete": False,
    }


@app.post("/api/messages")
async def send_message(
    request: Request,
    message: ApplicantMessage,
    background_tasks: BackgroundTasks,
):
    _, interview = get_interview(message.email)
    applicant_turns = sum(
        turn["role"] == "applicant" for turn in interview
    )
    max_turns = int(os.environ["INTERVIEW_MAX_TURNS"])
    if applicant_turns >= max_turns:
        raise HTTPException(status_code=409, detail="Interview complete")

    applicant_message = add_message(message.email, "applicant", message.content)
    interview_complete = applicant_turns + 1 >= max_turns
    if interview_complete:
        interviewer_message = add_message(
            message.email,
            "interviewer",
            "Thank you for your time. The interview is now complete.",
        )
    else:
        interviewer_message = await create_interviewer_message(
            request.app.state.openai,
            message.email,
        )
    background_tasks.add_task(
        request.app.state.judge_queue.put,
        ("message", message.email, applicant_message["id"]),
    )
    return {
        "applicant_message": applicant_message,
        "interviewer_message": interviewer_message,
        "interview_complete": interview_complete,
    }


@app.get("/api/reviewer/applications")
def applications():
    return list_applications()


@app.get("/api/reviewer/rubric")
def rubric():
    return load_rubric()


@app.get("/api/reviewer/applications/{email}")
def application(email: str):
    application_record, messages, belief_updates = get_application(email)
    if application_record is None:
        raise HTTPException(status_code=404)
    return {
        "application": application_record,
        "messages": messages,
        "belief_updates": belief_updates,
        "rubric": load_rubric(),
    }


@app.get("/api/reviewer/applications/{email}/resume")
def resume(email: str):
    resume_record = get_resume(email)
    if resume_record is None:
        raise HTTPException(status_code=404)
    return Response(
        content=resume_record["resume_data"],
        media_type=resume_record["resume_content_type"],
        headers={
            "Content-Disposition": (
                f'inline; filename="{resume_record["resume_filename"]}"'
            )
        },
    )


@app.get("/api/session")
def session(x_reviewer_username: str = Header()):
    return {"username": x_reviewer_username}
