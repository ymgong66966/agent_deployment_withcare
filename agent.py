import logging
import pickle
import asyncio
import uuid

import numpy as np
from typing import Annotated
from pathlib import Path
from datetime import datetime
import os
import snowflake.connector

from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    RoomInputOptions,
    AutoSubscribe
)
from livekit.plugins import deepgram, openai, silero, noise_cancellation
# from livekit.plugins.turn_detector.english import EnglishModel
import aiofiles
load_dotenv()

logger = logging.getLogger("rag-assistant")
logger.setLevel(logging.INFO)

# EMBEDDINGS_DIMENSION = 1536
# INDEX_PATH = "vdb_data"
# DATA_PATH = "my_data.pkl"

# Add chat context lock
_chat_ctx_lock = asyncio.Lock()

class SnowflakeConnectionPool:
    """Simple connection pool for Snowflake connections"""
    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self.connections = asyncio.Queue(maxsize=max_connections)
        self.created_connections = 0
        self.lock = asyncio.Lock()
        
    async def get_connection(self):
        """Get a connection from the pool"""
        try:
            # Try to get an existing connection
            return self.connections.get_nowait()
        except asyncio.QueueEmpty:
            # Create a new connection if under limit
            async with self.lock:
                if self.created_connections < self.max_connections:
                    conn = await self._create_connection()
                    if conn:
                        self.created_connections += 1
                        return conn
                    
            # Wait for a connection to be returned
            return await self.connections.get()
    
    async def return_connection(self, conn):
        """Return a connection to the pool"""
        if conn and not conn.is_closed():
            try:
                self.connections.put_nowait(conn)
            except asyncio.QueueFull:
                # Pool is full, close the connection
                conn.close()
                
    async def _create_connection(self):
        """Create a new Snowflake connection"""
        try:
            conn = await asyncio.to_thread(
                snowflake.connector.connect,
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                account=os.getenv('SNOWFLAKE_ACCOUNT')
            )
            
            # Set up database and schema
            cursor = conn.cursor()
            database = os.getenv('SNOWFLAKE_DATABASE')
            schema = os.getenv('SNOWFLAKE_SCHEMA')
            cursor.execute(f"USE DATABASE {database}")
            cursor.execute(f"USE SCHEMA {schema}")
            cursor.close()
            
            return conn
        except Exception as e:
            logger.error(f"Failed to create Snowflake connection: {e}")
            return None
    
    async def close_all(self):
        """Close all connections in the pool"""
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()
            except asyncio.QueueEmpty:
                break

# Global connection pool
snowflake_pool = SnowflakeConnectionPool(max_connections=3)


async def get_caregiver_profile():
    """Fetch a random caregiver profile from Snowflake database"""
    conn = None
    try:
        # Check if all required environment variables are set
        required_vars = ['SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 'SNOWFLAKE_ACCOUNT', 
                         'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA']
        
        for var in required_vars:
            if not os.getenv(var):
                logger.warning(f"Missing required environment variable: {var}")
                return "", -1
                
        # Get a connection from the pool
        conn = await snowflake_pool.get_connection()
        if not conn:
            return "", -1
        
        # Get a random caregiver profile
        cursor = conn.cursor()
        cursor.execute("""
        SELECT profile_string, profile_id 
        FROM CAREGIVER_PROFILES 
        ORDER BY RANDOM() 
        LIMIT 1
        """)
        
        result = cursor.fetchone()
        profile_string = result[0] if result else ""
        profile_id = result[1] if result else -1
        
        return profile_string, profile_id
        
    except Exception as e:
        logger.error(f"Error fetching caregiver profile from Snowflake: {e}")
        return "", -1
    finally:
        # Always return the connection to the pool
        if conn:
            await snowflake_pool.return_connection(conn)


async def save_chat_to_snowflake(chat_id: str, chat_turn: str, profile_id: int = None, chat_time: datetime = None, chat_type: str = None):
    """Save a chat message to Snowflake database"""
    try:
        # Check if all required environment variables are set
        required_vars = ['SNOWFLAKE_USER', 'SNOWFLAKE_PASSWORD', 'SNOWFLAKE_ACCOUNT', 
                         'SNOWFLAKE_DATABASE', 'SNOWFLAKE_SCHEMA']
        
        for var in required_vars:
            if not os.getenv(var):
                logger.warning(f"Missing required environment variable: {var}")
                return False
        
        # Get a connection from the pool
        conn = await snowflake_pool.get_connection()
        
        # Insert the chat message
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO CAREGIVER_CHAT (chat_id, chat_turn, profile_id, chat_time, chat_type)
        VALUES (%s, %s, %s, %s, %s)
        """, (chat_id, chat_turn, profile_id, chat_time or datetime.now(), chat_type))
        
        logger.debug(f"Successfully inserted message: {chat_turn[:50]}...")
        
        # Return the connection to the pool
        await snowflake_pool.return_connection(conn)
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving chat to Snowflake: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


class CaregiverAgent(Agent):
    """Custom agent class for caregiver persona"""
    def __init__(self, profile_text: str):
        instructions = f"""You are a caregiver that has the following background information or persona: {profile_text}  /n/n
#     Now you are having a phone call with a social worker, you want to talk to her in a human daily conversational way, Improvise when you are asked questions that are not covered by your background information but remember to make sure it is consistent with your persona. Your goal is to find a reliable and affordable care provider for your care receiver. You need figure it out by working with this social worker, asking her questions and answering her questions as well. The most important thing is to not break out of your persona when you are talking to the social worker. You are a human, not a computer or assistant. Another important thing is that you need to try your best to sounds like a human caregiver, which means that your sentences should be very oral and human-like, not like a computer or assistant. Your output doesn't have to be very formal and long. Your output doesn't have to be very formal and long. Most importantly, you need to speak in a style that is consistent with your persona described in your background information. Try to gradually give off information about your background information, don't just give it all at once. RESTRAIN your responses to the social worker to 1-2 sentences at a time!!!! Or else you will be fired. Very IMPORTANTLY, RESTRAIN your responses to the social worker to 1-2 sentences at a time!!!! Or else you will be fired. Very IMPORTANTLY, RESTRAIN your responses to the social worker to 1-2 sentences at a time!!!! Or else you will be fired. REMEMBER: you are kind of letting the socail worker you are talking to to figure out what you need gradually, even though you already know what you really need. You should not give away your need so blantly and easily. You should just start off by talking about your life and your feelings. Let the social worker engage gradually, and let them find out what you need. also, don't say something like : "how about you?", "what about you?", "how is your day going". You don't have to say things like that to be friendly or engage the social worker. The social worker is supposed to make conversations with you and care for you. Let them do more talking."""
        super().__init__(instructions=instructions)


async def entrypoint(ctx: JobContext):
    # Generate unique chat session ID
    chat_session_id = str(uuid.uuid4())
    logger.info(f"Starting new chat session: {chat_session_id}")
    
    # Fetch caregiver profile from Snowflake
    caregiver_profile, profile_id = await get_caregiver_profile()
    
    # Use the profile if available, otherwise use the default profile
    profile_text = caregiver_profile if caregiver_profile else """{"Name": "Kristine", "age":"36", "gender":"female","ethnicity":"Hispanic","Parents to be taken care of": "father", "Care receiver age": "67", "Care receiver background": "veteran, disabled, can't walk, on a wheelchair, lonely and needs company, speaks only Spanish, lives in chicago, streeterville, 60611", "Care receiver insurance": "United". "caregiver background":"need to work Monday to Saturday, 8 am to 8 pm, don't have anyone to take care of father. Live 1 hr away from him. It is been stressful taking care of father."}"""
    
    # Connect to the room with auto-subscription to audio
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create an agent session with the necessary components
    session = AgentSession(
        # turn_detection=EnglishModel(),
        stt=deepgram.STT(model="nova-3", language="en"),
        vad=silero.VAD.load(activation_threshold=0.8,  # Higher threshold = less sensitive (default: 0.5)
            min_silence_duration=0.8,  # Longer silence needed to stop (default: 0.55)
            min_speech_duration=0.3, ),
        llm=openai.LLM(),
        tts=openai.TTS(),
        allow_interruptions=True,
        min_interruption_duration=0.7,
    )
    
    # Create the agent with the caregiver profile
    agent = CaregiverAgent(profile_text)
    
    # Start the session with the agent
    # In Agents 1.0, text input is automatically handled through text streams on lk.chat topic
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    
    log_queue = asyncio.Queue()

    # Handle user input transcription events
    # @session.on("user_input_transcribed")
    # def on_user_input_transcribed(event):
    #     print(f"USER TRANSCRIBED: {event.transcript} (final: {event.is_final})")
    #     logger.debug(f"User input transcribed: {event.transcript[:50]}... (final: {event.is_final})")
        # Only log final transcripts to avoid duplicates
        # if event.is_final and len(event.transcript.strip()) > 2:
        #     log_queue.put_nowait({
        #         "type": "USER_FINAL",
        #         "content": event.transcript,
        #         "timestamp": datetime.now()
        #     })

    # Handle conversation items (both user and agent messages)
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        item = event.item
        print(f"CONVERSATION ITEM: {item.role.upper()} - {item.text_content}")
        logger.debug(f"Conversation item added from {item.role}: {item.text_content[:50]}...")
        
        if len(item.text_content.strip()) > 10:
            log_queue.put_nowait({
                "type": f"{item.role.upper()}_CONVERSATION",
                "content": item.text_content,
                "timestamp": datetime.now()
            })

    # Handle agent speech creation
    # @session.on("speech_created")
    # def on_speech_created(event):
    #     print(f"AGENT SPEECH CREATED: source={event.source}, user_initiated={event.user_initiated}")
    #     logger.debug(f"Agent speech created: source={event.source}, user_initiated={event.user_initiated}")
        
    #     # We can track when speech is created, but the actual content will come through conversation_item_added
    #     log_queue.put_nowait({
    #         "type": "AGENT_SPEECH_CREATED",
    #         "content": f"Speech created - source: {event.source}, user_initiated: {event.user_initiated}",
    #         "timestamp": datetime.now()
    #     })
    


    # Add a general event listener to see what events are being fired
    def log_all_events(event_name, *args, **kwargs):
        logger.debug(f"Event fired: {event_name} with args: {args[:2]}...")  # Limit args to avoid spam
    
    # Try to catch common events that might exist
    # event_names = [
    #     "user_input_transcribed", "conversation_item_added", "speech_created",
    #     # "function_tools_executed", "metrics_collected", "agent_state_changed", "user_state_changed"
    # ]
    
    # for event_name in event_names:
    #     try:
    #         session.on(event_name, lambda *args, name=event_name, **kwargs: log_all_events(name, *args, **kwargs))
    #     except Exception as e:
    #         logger.debug(f"Could not register handler for {event_name}: {e}")

    async def save_chat_messages(profile_id: int):
        """Process messages from queue and save to Snowflake"""
        while True:
            try:
                msg_data = await log_queue.get()
                if msg_data is None:
                    break
                
                # Format message for database
                formatted_message = msg_data['content']
                
                # Save to Snowflake (profile_id can be extracted from profile_text if needed)
                success = await save_chat_to_snowflake(
                    chat_id=chat_session_id,
                    chat_turn=formatted_message,
                    profile_id=profile_id,
                    chat_time=msg_data['timestamp'],
                    chat_type=msg_data['type']
                )
                
                if success:
                    logger.info(f"Saved message to Snowflake: {msg_data['type']}")
                else:
                    logger.error(f"Failed to save message to Snowflake: {msg_data['type']}")
                    
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    save_task = asyncio.create_task(save_chat_messages(profile_id))

    async def finish_queue():
        log_queue.put_nowait(None)
        await save_task
        # Close all Snowflake connections
        await snowflake_pool.close_all()

    ctx.add_shutdown_callback(finish_queue)

    # # Log session start
    # log_queue.put_nowait({
    #     "type": "SYSTEM",
    #     "content": f"Agent started and ready for conversation. Session ID: {chat_session_id}",
    #     "timestamp": datetime.now()
    # })
    
    # Send initial greeting
    await session.generate_reply(instructions="""say hello to the user and start the conversation. Remember, you are a human caregiver. you need to try your best to sounds like a human caregiver, which means that your sentences should be very oral and human-like, not like a computer or assistant. Your output doesn't have to be very formal and long. You don't have to tell all about your situation. You just need to start the conversation simple and to the point. Most importantly, you need to speak in a style that is consistent with your persona described in your background information. Try to gradually give off information about your background information, don't just give it all at once. Very IMPORTANTLY, RESTRAIN your responses to the social worker to 1-2 sentences at a time!!!! Or else you will be fired. Very IMPORTANTLY, RESTRAIN your responses to the social worker to 1-2 sentences at a time!!!! Or else you will be fired. Very IMPORTANTLY, RESTRAIN your responses to the social worker to 1-2 sentences at a time!!!! Or else you will be fired. REMEMBER: you are kind of letting the socail worker you are talking to to figure out what you need gradually, even though you already know what you really need. You should not give away your need so blantly and easily. You should just start off by talking about your life and your feelings. Let the social worker engage gradually, and let them find out what you need. also, don't say something like : "how about you?", "what about you?". You don't have to say things like that to be friendly or engage the social worker. The social worker is supposed to make conversations with you. Let them do more talking.""")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))