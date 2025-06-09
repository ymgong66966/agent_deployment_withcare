import logging
import pickle
import asyncio

import numpy as np
from typing import Annotated
from pathlib import Path
from livekit import rtc
from datetime import datetime
import os
import snowflake.connector

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero
import aiofiles
load_dotenv()

logger = logging.getLogger("rag-assistant")
logger.setLevel(logging.INFO)
# EMBEDDINGS_DIMENSION = 1536
# INDEX_PATH = "vdb_data"
# DATA_PATH = "my_data.pkl"

# Add chat context lock
_chat_ctx_lock = asyncio.Lock()


async def get_caregiver_profile():
    """Fetch the latest caregiver profile from Snowflake database"""
    try:
        # Connect to Snowflake
        ctx = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT')
        )
        cursor = ctx.cursor()
        
        # Set the database and schema
        database = os.getenv('SNOWFLAKE_DATABASE')
        schema = os.getenv('SNOWFLAKE_SCHEMA')
        
        # Use the specified database and schema
        cursor.execute(f"USE DATABASE {database}")
        cursor.execute(f"USE SCHEMA {schema}")
        
        # Get the latest profile
        cursor.execute("""
        SELECT profile_string 
        FROM CAREGIVER_PROFILES 
        ORDER BY created_at DESC 
        LIMIT 1
        """)
        
        result = cursor.fetchone()
        profile_string = result[0] if result else ""
        
        # Close the connection
        cursor.close()
        ctx.close()
        
        return profile_string
    except Exception as e:
        logger.error(f"Error fetching caregiver profile from Snowflake: {e}")
        return ""


async def entrypoint(ctx: JobContext):
    # Fetch caregiver profile from Snowflake
    caregiver_profile = await get_caregiver_profile()
    
    # Use the profile if available, otherwise use the default profile
    profile_text = caregiver_profile if caregiver_profile else """{"Name": "Kristine", "age":"36", "gender":"female","ethnicity":"Hispanic","Parents to be taken care of": "father", "Care receiver age": "67", "Care receiver background": "veteran, disabled, can't walk, on a wheelchair, lonely and needs company, speaks only Spanish, lives in chicago, streeterville, 60611", "Care receiver insurance": "United". "caregiver background":"need to work Monday to Saturday, 8 am to 8 pm, don't have anyone to take care of father. Live 1 hr away from him. It is been stressful taking care of father."}"""
    
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            f"""You are a caregiver that has the following background information or persona: {profile_text}  /n/n
#     Now you are having a phone call with a social worker, you want to talk to her in a human daily conversational way, Improvise when you are asked questions that are not covered by your background information but remember to make sure it is consistent with your persona. Your goal is to find a reliable and affordable care provider for your care receiver. You need figure it out by working with this social worker, asking her questions and answering her questions as well. The most important thing is to not break out of your persona when you are talking to the social worker. You are a human, not a computer or assistant. """
        ),
    )

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        chat_ctx=initial_ctx,
    )
    agent.start(ctx.room)

    # listen to incoming chat messages, only required if you'd like the agent to
    # answer incoming messages from Chat
    chat = rtc.ChatManager(ctx.room)

    async def answer_from_text(txt: str):
        chat_ctx = agent.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        stream = agent.llm.chat(chat_ctx=chat_ctx)
        await agent.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    log_queue = asyncio.Queue()

    @agent.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        # convert string lists to strings, drop images
        if isinstance(msg.content, list):
            msg.content = "\n".join(
                "[image]" if isinstance(x, llm.ChatImage) else x for x in msg
            )
        print(msg.content)
        log_queue.put_nowait(f"[{datetime.now()}] USER:\n{msg.content}\n\n")

    @agent.on("agent_speech_committed")
    def on_agent_speech_committed(msg: llm.ChatMessage):
        print(msg.content)
        log_queue.put_nowait(f"[{datetime.now()}] AGENT:\n{msg.content}\n\n")

    async def write_transcription():
        async with aiofiles.open("text.txt", "a") as f:
            while True:
                msg = await log_queue.get()
                if msg is None:
                    break
                await f.write(msg)
                # await f.flush() 
    write_task = asyncio.create_task(write_transcription())

    async def finish_queue():
        log_queue.put_nowait(None)
        await write_task

    ctx.add_shutdown_callback(finish_queue)
    log_queue.put_nowait(f"[{datetime.now()}] SYSTEM: Agent started and ready for conversation\n\n")
    await agent.say("Hey, nice to talk to you!", allow_interruptions=True)






if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))