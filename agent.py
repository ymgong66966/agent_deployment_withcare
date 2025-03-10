import logging
import pickle
import asyncio
import random
import wave
import numpy as np
from typing import Annotated
from pathlib import Path
from livekit import rtc
from datetime import datetime
import os
from snowflake import connector

from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, rag, silero
import aiofiles
load_dotenv()

logger = logging.getLogger("rag-assistant")
logger.setLevel(logging.INFO)
# EMBEDDINGS_DIMENSION = 1536
# INDEX_PATH = "vdb_data"
# DATA_PATH = "my_data.pkl"

# Add chat context lock
_chat_ctx_lock = asyncio.Lock()

# annoy_index = rag.annoy.AnnoyIndex.load(INDEX_PATH)
# with open(DATA_PATH, "rb") as f:
#     paragraphs_by_uuid = pickle.load(f)

# # Cache for wav file data to avoid repeated disk reads
# _wav_cache = {}
# # Cache for audio track and source
# _wav_audio_track = None
# _wav_audio_source = None



# def get_snowflake_connection():
#     """Create and return a Snowflake connection"""
#     return connector.connect(
#     user='ymgong',
#     password='ZhangShouHua66966',
#     account='dxmfoqi-gl49063',
#     database='WITHCARE_TRANSCRIPTS',
#     schema='PUBLIC',
#     warehouse='COMPUTE_WH'
# )

# async def save_transcription(speaker: str, text: str, timestamp: datetime):
#     """Save transcription to Snowflake"""
#     try:
#         conn = get_snowflake_connection()
#         cursor = conn.cursor()
        
#         # Create table if it doesn't exist
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS call_transcriptions (
#             id NUMBER AUTOINCREMENT,
#             speaker VARCHAR,
#             text VARCHAR,
#             timestamp TIMESTAMP_NTZ,
#             created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
#         )
#         """)
        
#         # Insert transcription
#         cursor.execute(
#             "INSERT INTO call_transcriptions (speaker, text, timestamp) VALUES (%s, %s, %s)",
#             (speaker, text, timestamp)
#         )
        
#         conn.commit()
#     except Exception as e:
#         logger.error(f"Error saving transcription to Snowflake: {e}")
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'conn' in locals():
#             conn.close()

# class TranscriptionCapturingAgent(VoicePipelineAgent):
#     """Extended agent that captures transcriptions"""
    
#     async def on_transcript(self, text: str, is_final: bool):
#         """Override to capture transcriptions"""
#         if is_final:
#             await save_transcription(
#                 speaker="user",
#                 text=text,
#                 timestamp=datetime.utcnow()
#             )
#         return await super().on_transcript(text, is_final)
    
    # async def say(self, text_stream, add_to_chat_ctx=True, **kwargs):
    #     """Override to capture agent responses"""
    #     # Handle string input
    #     if isinstance(text_stream, str):
    #         response = await super().say(text_stream, add_to_chat_ctx=add_to_chat_ctx, **kwargs)
    #         # Save the complete response
    #         if response:
    #             await save_transcription(
    #                 speaker="agent",
    #                 text=response,
    #                 timestamp=datetime.utcnow()
    #             )
    #         yield response
    #     else:
    #         # For async generators, collect chunks while yielding
    #         full_response = ""
    #         async for chunk in text_stream:
    #             full_response += chunk
    #             yield chunk
            
    #         # Save the complete response
    #         if full_response:
    #             await save_transcription(
    #                 speaker="agent",
    #                 text=full_response,
    #                 timestamp=datetime.utcnow()
    #             )
    #         yield full_response

# async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext) -> None:
#     """
#     Locate the last user message, use it to query the RAG model for
#     the most relevant paragraph, add that to context, and generate a response.
#     """
#     async with _chat_ctx_lock:
#         user_msg = chat_ctx.messages[-1]

#     # Let's sleep for 5 seconds to simulate a delay
#     await asyncio.sleep(5)

#     user_embedding = await openai.create_embeddings(
#         input=[user_msg.content],
#         model="text-embedding-3-small",
#         dimensions=EMBEDDINGS_DIMENSION,
#     )

#     result = annoy_index.query(user_embedding[0].embedding, n=1)[0]
#     paragraph = paragraphs_by_uuid[result.userdata]

#     if paragraph:
#         logger.info(f"enriching with RAG: {paragraph}")
#         rag_msg = llm.ChatMessage.create(
#             text="Context:\n" + paragraph,
#             role="assistant",
#         )
        
#         async with _chat_ctx_lock:
#             # Replace last message with RAG, then append user message at the end
#             chat_ctx.messages[-1] = rag_msg
#             chat_ctx.messages.append(user_msg)

#             # Generate a response using the enriched context
#             llm_stream = agent._llm.chat(chat_ctx=chat_ctx)
#             await agent.say(llm_stream)

# async def play_wav_once(wav_path: str | Path, room: rtc.Room, volume: float = 0.3):
#     """
#     Simple function to play a WAV file once through a LiveKit audio track
#     This is only needed for the "Option 3" thinking message in the entrypoint function.
#     """
#     global _wav_audio_track, _wav_audio_source
#     samples_per_channel = 9600
#     wav_path = Path(wav_path)
    
#     # Create audio source and track if they don't exist
#     if _wav_audio_track is None:
#         _wav_audio_source = rtc.AudioSource(48000, 1)
#         _wav_audio_track = rtc.LocalAudioTrack.create_audio_track("wav_audio", _wav_audio_source)
        
#         # Only publish the track once
#         await room.local_participant.publish_track(
#             _wav_audio_track,
#             rtc.TrackPublishOptions(
#                 source=rtc.TrackSource.SOURCE_MICROPHONE,
#                 stream="wav_audio"
#             )
#         )
        
#         # Small delay to ensure track is established
#         await asyncio.sleep(0.5)
    
#     try:
#         # Use cached audio data if available
#         if str(wav_path) not in _wav_cache:
#             with wave.open(str(wav_path), 'rb') as wav_file:
#                 audio_data = wav_file.readframes(wav_file.getnframes())
#                 audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                
#                 if wav_file.getnchannels() == 2:
#                     audio_array = audio_array.reshape(-1, 2).mean(axis=1)
                
#                 _wav_cache[str(wav_path)] = audio_array
        
#         audio_array = _wav_cache[str(wav_path)]
        
#         for i in range(0, len(audio_array), samples_per_channel):
#             chunk = audio_array[i:i + samples_per_channel]
            
#             if len(chunk) < samples_per_channel:
#                 chunk = np.pad(chunk, (0, samples_per_channel - len(chunk)))
            
#             chunk = np.tanh(chunk / 32768.0) * 32768.0
#             chunk = np.round(chunk * volume).astype(np.int16)
            
#             await _wav_audio_source.capture_frame(rtc.AudioFrame(
#                 data=chunk.tobytes(),
#                 sample_rate=48000,
#                 samples_per_channel=samples_per_channel,
#                 num_channels=1
#             ))
            
#             await asyncio.sleep((samples_per_channel / 48000) * 0.98)
#     except Exception as e:
#         # If something goes wrong, clean up the track and source so they can be recreated
#         if _wav_audio_track:
#             await _wav_audio_track.stop()
#             await room.local_participant.unpublish_track(_wav_audio_track)
#         if _wav_audio_source:
#             _wav_audio_source.close()
#         _wav_audio_track = None
#         _wav_audio_source = None
#         raise e


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            """You are a caregiver that has the following background information or persona: {"Name": "Kristine", "age":"36", "gender":"female","ethnicity":"Hispanic","Parents to be taken care of": "father", "Care receiver age": "67", "Care receiver background": "veteran, disabled, can't walk, on a wheelchair, lonely and needs company, speaks only Spanish, lives in chicago, streeterville, 60611", "Care receiver insurance": "United". "caregiver background":"need to work Monday to Saturday, 8 am to 8 pm, don't have anyone to take care of father. Live 1 hr away from him. It is been stressful taking care of father."}  /n/n
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
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)





# async def entrypoint(ctx: JobContext) -> None:
#     """
#     Main entrypoint for the agent. Sets up function context, defines
#     RAG enrichment command, creates the agent's initial conversation context,
#     and starts the agent.
#     """
#     fnc_ctx = llm.FunctionContext()

#     agent = TranscriptionCapturingAgent(
#         chat_ctx=llm.ChatContext().append(
#             role="system",
#             text=(
#                 """You are a caregiver that has the following background information or persona: {"Name": "Kristine", "age":"36", "gender":"female","ethnicity":"Hispanic","Parents to be taken care of": "father", "Care receiver age": "67", "Care receiver background": "veteran, disabled, can't walk, on a wheelchair, lonely and needs company, speaks only Spanish, lives in chicago, streeterville, 60611", "Care receiver insurance": "United". "caregiver background":"need to work Monday to Saturday, 8 am to 8 pm, don't have anyone to take care of father. Live 1 hr away from him. It is been stressful taking care of father."}  /n/n
#     Now you are having a phone call with a social worker, you want to talk to her in a human daily conversational way, Improvise when you are asked questions that are not covered by your background information but remember to make sure it is consistent with your persona. Your goal is to find a reliable and affordable care provider for your care receiver. You need figure it out by working with this social worker, asking her questions and answering her questions as well. The most important thing is to not break out of your persona when you are talking to the social worker. You are a human, not a computer or assistant. """
#                 # "Option 1: Include this in the system prompt to make the agent say that it's looking up the answer w/ every function call. This doesn't always work, but is the simplest solution."
#                 # "If you need to perform a function call, always tell the user that you are looking up the answer."
#             ),
#         ),
#         vad=silero.VAD.load(),
#         stt=deepgram.STT(),
#         llm=openai.LLM(),
#         tts=openai.TTS(),
#         fnc_ctx=fnc_ctx,
#     )

#     @fnc_ctx.ai_callable()
#     async def enrich_with_rag(
#         code: Annotated[
#             int, llm.TypeInfo(description="Enrich with RAG for questions about LiveKit.")
#         ]
#     ):
#         """
#         Called when you need to enrich with RAG for questions about LiveKit.
#         """
#         logger.info("Enriching with RAG for questions about LiveKit")

#         ############################################################
#         # Options for thinking messages
#         # Option 1 is included in the system prompt
#         ############################################################

#         # Option 2: Use a message from a specific list to indicate that we're looking up the answer
#         thinking_messages = [
#             "Let me look that up...",
#             "One moment while I check...",
#             "I'll find that information for you...",
#             "Just a second while I search...",
#             "Looking into that now..."
#         ]
#         await agent.say(random.choice(thinking_messages))

#         # Option 3: Make a call to the LLM to generate a custom message for this specific function call
#         # async with _chat_ctx_lock:
#         #     thinking_ctx = llm.ChatContext().append(
#         #         role="system",
#         #         text="Generate a very short message to indicate that we're looking up the answer in the docs"
#         #     )
#         #     thinking_stream = agent._llm.chat(chat_ctx=thinking_ctx)
#         #     # Wait for thinking message to complete before proceeding
#         #     await agent.say(thinking_stream, add_to_chat_ctx=False)

#         # Option 4: Play an audio file through the room's audio track
#         # await play_wav_once("let_me_check_that.wav", ctx.room)

#         ############################################################
#         ############################################################

#         await _enrich_with_rag(agent, agent.chat_ctx)

#     # Connect and start the agent
#     await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

#     agent.start(ctx.room)
#     # Now we can await the say method directly
#     await agent.say("Hey I am calling for help!", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))