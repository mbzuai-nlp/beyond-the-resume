import chainlit as cl
import uuid
import asyncio
from engine import engine

AGENT_KEY = "agent"
JUDGEMENT_HISTORY_KEY = "judgement_history"


async def refresh_rubric_sidebar(judgement_history: list):
    widget = cl.CustomElement(
        name="RubricWidget",
        props={"judgements": {"history": judgement_history}},
    )

    await cl.ElementSidebar.set_title("Belief")
    await cl.ElementSidebar.set_elements([widget], key=f"rubric-{uuid.uuid4()}")


async def thinking_spinner(message: cl.Message, interval: float = 0.6):
    dots = 0
    try:
        while True:
            suffix = "." * dots
            message.content = f"Thinking (this can take up to 60s){suffix}"
            await message.update()
            dots = (dots + 1) % 4  # "", ".", "..", "..."
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        return


@cl.on_chat_start
async def start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Hi there, and thank you for your interest in joining Big Data Group. Please upload your resume to begin!",
            accept=["application/pdf"],
        ).send()

    text_file = files[0]
    agent = engine.construct_next_message(text_file.path)
    cl.user_session.set(AGENT_KEY, agent)

    # init history
    cl.user_session.set(JUDGEMENT_HISTORY_KEY, [])

    status = cl.Message(content="Thinking (this can take up to 60s)")
    await status.send()

    spinner_task = asyncio.create_task(thinking_spinner(status, interval=0.6))
    try:
        first_msg, judgement = await anext(agent)
    finally:
        spinner_task.cancel()

    status.content = first_msg
    await status.update()

    hist = cl.user_session.get(JUDGEMENT_HISTORY_KEY) or []
    hist.append(judgement)
    cl.user_session.set(JUDGEMENT_HISTORY_KEY, hist)

    await refresh_rubric_sidebar(hist)


@cl.on_message
async def on_message(msg: cl.Message):
    agent = cl.user_session.get(AGENT_KEY)
    if not agent:
        await cl.Message("Agent has exited. Please create a new conversation.").send()
        return

    status = cl.Message(content="Thinking (this can take up to 60s)")
    await status.send()

    agent_task = asyncio.create_task(agent.asend(msg.content))
    spinner_task = asyncio.create_task(thinking_spinner(status, interval=0.6))

    try:
        next_msg, judgement = await agent_task
    finally:
        spinner_task.cancel()
        try:
            await spinner_task
        except asyncio.CancelledError:
            pass

    status.content = next_msg
    await status.update()

    hist = cl.user_session.get(JUDGEMENT_HISTORY_KEY) or []
    hist.append(judgement)
    cl.user_session.set(JUDGEMENT_HISTORY_KEY, hist)

    await refresh_rubric_sidebar(hist)


@cl.on_chat_end
async def end():
    print("CHAT UNEXPECTEDLY ENDED.")