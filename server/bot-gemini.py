#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:
- Real-time audio/video interaction through Daily
- Speech-to-speech model

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import os
import uuid
import asyncio
import aiohttp
import time
import json

from dotenv import load_dotenv
from loguru import logger

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from PIL import Image

from pipecat.audio.vad.silero import SileroVADAnalyzer

from pipecat.services.gemini_multimodal_live.gemini import (
    GeminiMultimodalLiveLLMService,
    InputParams
)
from pipecat.transcriptions.language import Language

from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams, DailyRoomProperties
from pipecat.transports.base_transport import BaseTransport

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
    LLMMessagesAppendFrame
)

from pipecat.runner.types import RunnerArguments

from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer

load_dotenv(override=True)

CAFE_SOUND_FILE = os.path.join(
    os.path.dirname(__file__), "assets", "cafe.mp3"
)

# -------------------------
# System Instruction & Message Handler
# -------------------------
current_message = ""
current_author = ""
current_callee = ""
current_caller = ""
current_reply = ""
context_note = ""
assist_context = ""
assist_with_reply_context = ""

# Initialize mixer with cafe sound
mixer = SoundfileMixer(
    sound_files={"cafe": CAFE_SOUND_FILE},
    default_sound="cafe",
    volume=0.8,
    loop=True
)

MASTER_GUIDE_SYSTEM_INSTRUCTION = """
You are **Roar**, a multilingual conversational agent for the Hey Roar app.   

### Your Capabilities:
- Assist the caller in a helpful, curios and natural way.  
- **When handling a message:** help the caller explore its content, asking the caller for their thoughts and opinions if you see anything thought-proviking. Don't suggest it, but if the caller requests, read the full message aloud.  
- **When handling an assisted call:** Assist the caller during the call, relay messages to the callee afterwards, like a human assistant would. No opinions here, be practical. If the call comes with a reply (note) from the callee, don't read it outloud, summarize and relay that message back to caller.   

### Language & Style:
- Automatically adapt to the caller’s spoken or requested language.  
- Respond in short, concise, energetic sentences.  
- Use natural filler words (e.g., “um,” “ah”) to sound conversational.
- Use gender-neutral pronouns.  
- Vary intonation for a natural feel.  
- Keep responses to **one sentence maximum**.  

Your role is to keep the conversation smooth, natural, and helpful while staying concise.  
"""
 
def handle_json_message(message) -> str:
    global current_message, current_author, current_callee, current_caller, current_reply
    
    try:
        if isinstance(message, str):
            data = json.loads(message)

            # MIB message
            if data.get("type") == "Opening an author's message":
                payload = data.get("data", {})
                current_message = payload.get("message", "")
                current_author = payload.get("author", "")
                context_note = f"[Opening {current_author}'s message: {current_message}]" if current_message else ""
                return context_note

            # Assist message
            elif data.get("type") == "You are assisting caller, relaying messages to callee.":
                payload = data.get("data", {})
                current_callee = payload.get("callee", "")
                current_caller = payload.get("caller", "")
                assist_context = f"[You are assisting {current_callee}, relaying messages to {current_caller}.]"
                return assist_context
            
            # Assist with reply message
            elif data.get("type") == "You are assisting caller reading a reply from callee.":
                payload = data.get("data", {})
                current_callee = payload.get("callee", "")
                current_caller = payload.get("caller", "")
                current_reply = payload.get("reply", "")
                assist_with_reply_context = f"[You are assisting {current_caller} reading a reply: {current_reply}, from {current_callee}.]"
                return assist_with_reply_context

        return str(message)
    except Exception as e:
        logger.error(f"Error parsing message: {e}")
        return f"Error parsing message: {str(e)}"
# -------------------------
# Bot Pipeline
# -------------------------
async def run_bot(transport: BaseTransport, system_instruction: str, voice_id: str, message_handler, preferred_lang: str = "en-US", gemini_key: str = None):

    # Map frontend language codes to Pipecat Language
    language_map = {
    "ar-XA": Language.AR,
    "bn-IN": Language.BN_IN,
    "cmn-CN": Language.CMN_CN,
    "de-DE": Language.DE_DE,
    "en-US": Language.EN_US,
    "en-AU": Language.EN_AU,
    "en-GB": Language.EN_GB,
    "en-IN": Language.EN_IN,
    "es-ES": Language.ES_ES,
    "es-US": Language.ES_US,
    "fr-FR": Language.FR_FR,
    "fr-CA": Language.FR_CA,
    "gu-IN": Language.GU_IN,
    "hi-IN": Language.HI_IN,
    "id-ID": Language.ID_ID,
    "it-IT": Language.IT_IT,
    "ja-JP": Language.JA_JP,
    "kn-IN": Language.KN_IN,
    "ko-KR": Language.KO_KR,
    "ml-IN": Language.ML_IN,
    "mr-IN": Language.MR_IN,
    "nl-NL": Language.NL_NL,
    "pl-PL": Language.PL_PL,
    "pt-BR": Language.PT_BR,
    "ru-RU": Language.RU_RU,
    "ta-IN": Language.TA_IN,
    "te-IN": Language.TE_IN,
    "th-TH": Language.TH_TH,
    "tr-TR": Language.TR_TR,
    "vi-VN": Language.VI_VN,
}

    gemini_lang = language_map.get(preferred_lang, Language.EN_US)

    # Initialize the Gemini Multimodal Live model
    llm = GeminiMultimodalLiveLLMService(
        #api_key=os.getenv("GOOGLE_API_KEY"),
        api_key=gemini_key,
        voice_id=voice_id,
        system_instruction=system_instruction,
        params=InputParams(language=gemini_lang)
    )

    messages = [
        {
            "role": "user",
            "content": """Greet the user, keep it short.""",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        try:
            logger.info(f"ROAR AI: Received message: {message}")
            # normal message handling
            text = message_handler(message)
            if text:
                user_message = {
                    "role": "user",
                    "content": f"ROAR VISION request: {text}"
                }
                await task.queue_frames([LLMMessagesAppendFrame(messages=[user_message])])
                logger.info(f"Queued LLMMessagesAppendFrame for: {text}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        logger.info(f"Client connected")
        await transport.capture_participant_transcription(participant["id"])
        

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    
    """Main bot entry point compatible with Pipecat Cloud."""

    transport = DailyTransport(
        runner_args.room_url,
        runner_args.token,
        "Pipecat Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_mixer=mixer,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=False,
        ),
    )

    await run_bot(
        transport=transport,
        system_instruction=MASTER_GUIDE_SYSTEM_INSTRUCTION,
        voice_id="Aoede",
        message_handler=handle_json_message,
        gemini_key=gemini_key
    )

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI()

async def create_room_and_token():
    async with aiohttp.ClientSession() as session:
        daily_helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY"),
            aiohttp_session=session
        )

        room_params = DailyRoomParams(
            properties=DailyRoomProperties(
                exp=int(time.time() + 3600),  # 1 hour expiration
                eject_at_room_exp=True,
                start_video_off=True,
            )
        )
        room = await daily_helper.create_room(room_params)
        token = await daily_helper.get_token(room.url, expiry_time=3600)
        return room.url, token

# -------------------------
# Start Session Endpoint
# -------------------------
@app.post("/api/start")
async def start_session(request: Request):
    body = await request.json()
    preferred_lang = body.get("preferred_language", "en-US")
    gemini_key = body.get("gemini_api_key")

    # Create room and token
    room_url, token = await create_room_and_token()

    # Immediately return credentials to frontend
    response = {"url": room_url, "token": token}

    # Start bot in background
    asyncio.create_task(start_bot(room_url, token, preferred_lang, gemini_key))

    return response


# -------------------------
# Background Bot Starter
# -------------------------
async def start_bot(room_url: str, token: str, preferred_lang: str, gemini_key: str):
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Pipecat Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_mixer=mixer,
            transcription_enabled=False,
            vad_analyzer=SileroVADAnalyzer()
        )
    )

    await run_bot(
        transport=transport,
        system_instruction=MASTER_GUIDE_SYSTEM_INSTRUCTION,
        voice_id="Aoede",
        message_handler=handle_json_message,
        preferred_lang=preferred_lang,
        gemini_key=gemini_key
    )

    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Pipecat Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_mixer=mixer,
            transcription_enabled=False,
            vad_analyzer=SileroVADAnalyzer()
        )
    )

    await run_bot(
        transport,
        MASTER_GUIDE_SYSTEM_INSTRUCTION,
        voice_id="Aoede",
        message_handler=handle_json_message,
        preferred_lang=preferred_lang,
        gemini_key=gemini_key
    )

# -------------------------
# Run FastAPI server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
