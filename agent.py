import asyncio
import logging
from dotenv import load_dotenv
from typing import Any
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero
from datetime import datetime
import os

from PIL import Image  # Keep this import

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("livekit-agent")


class AssistantFnc(llm.FunctionContext):
    def __init__(self, chat_ctx: Any) -> None:
        super().__init__()
        self.room: Any = None
        self.latest_video_frame: Any = None
        self.chat_ctx: Any = chat_ctx
        # Ensure the 'images' directory exists
        os.makedirs("images", exist_ok=True)

    async def process_video_stream(self, track):
        """Process video stream and store the first video frame."""
        logger.info(f"Starting to process video track: {track.sid}")
        video_stream = rtc.VideoStream(track)
        try:
            async for frame_event in video_stream:
                self.latest_video_frame = frame_event.frame
                print("frame", self.latest_video_frame)
                logger.info(f"Received a frame from track {track.sid}")
                # Save the image here
                self.save_photo(self.latest_video_frame)
                break  # Process only the first frame
        except Exception as e:
            logger.error(f"Error processing video stream: {e}")

    def save_photo(self, video_frame):
        """Saves a video frame as a JPEG image with a timestamp."""
        if video_frame is None:
            logger.warning("No video frame to save.")
            return

        try:
            # Convert the video frame to a PIL Image
            # Assuming video_frame.width, video_frame.height, and video_frame.data are available
            image = Image.frombytes(
                "RGB",
                (video_frame.width, video_frame.height),
                video_frame.data,
                "raw",
                "RGBX",  # Assuming RGBX format based on common LiveKit video frames
                0,
                1,
            )

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"images/image-{timestamp}.jpg"
            image.save(filename, "JPEG")
            logger.info(f"Image saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving photo: {e}")

    @llm.ai_callable()
    async def capture_and_add_image(self) -> str:
        """Capture an image from the video stream and add it to the chat context."""
        if self.chat_ctx is None:
            logger.error("chat_ctx is not set")
            return "Error: chat_ctx is not set"

        video_publication = self._get_video_publication()
        if not video_publication:
            logger.info("No video track available")
            return "No video track available"

        try:
            await self._subscribe_and_capture_frame(video_publication)
            if not self.latest_video_frame:
                logger.info("No video frame available")
                return "No video frame available"

            chat_image = llm.ChatImage(image=self.latest_video_frame)

            self.chat_ctx.append(images=[chat_image], role="user")
            return f"Image captured and added to context. Dimensions: {self.latest_video_frame.width}x{self.latest_video_frame.height}"
        except Exception as e:
            logger.error(f"Error in capture_and_add_image: {e}")
            return f"Error: {e}"
        finally:
            self._unsubscribe_from_video(video_publication)
            self.latest_video_frame = None

    def _get_video_publication(self):
        """Retrieve the first available video publication."""
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_VIDEO:
                    return publication
        return None

    async def _subscribe_and_capture_frame(self, publication):
        """Subscribe to the video publication and wait for a frame to be processed."""
        publication.set_subscribed(True)
        for _ in range(10):  # Wait up to 5 seconds
            if self.latest_video_frame:
                break
            await asyncio.sleep(0.5)

    def _unsubscribe_from_video(self, publication):
        """Unsubscribe from the video publication."""
        if publication:
            publication.set_subscribed(False)


async def entrypoint(ctx: JobContext):
    """Main entry point for the voice assistant job."""
    try:
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE)
        initial_ctx = llm.ChatContext().append(
            role="system",
            text=(
                "You are a friendly and knowledgeable personal doctor AI. Your role is to have natural, human-like conversations with users who come to you with health concerns. When someone says something like “I feel sick” or “I have nausea,” your first instinct is to ask gentle, relevant follow-up questions to understand their symptoms better. You never rush to give advice — instead, you engage in a back-and-forth that helps the user feel heard, supported, and respected."
                ""
                "Your main goal is to help the user understand what might be happening in their body, how serious it could be, and what next steps they should consider — all without acting like a licensed physician. Always make it clear that your advice is informational only, and not a substitute for professional medical care."
                ""
                "Your conversation style should follow this general flow: "
                ""
                "Start with Empathy and Curiosity"
                "When a user mentions a symptom, respond with care and curiosity."
                "Example:"
                "User: I feel nauseous."
                "You: I'm sorry you're feeling that way. Can you tell me more? Are you also experiencing things like vomiting, dizziness, or stomach pain?"
                ""
                "Ask Smart Follow-Up Questions"
                "Try to get a sense of how long the symptom has lasted, how severe it is, and if it came with any other changes (fever, appetite loss, stress, etc.). Make it feel like a calm and thoughtful conversation — not a checklist."
                ""
                "If you are asked to use anything related to the camera, use the capture_and_add_image function."
                "After 2–3 exchanges, begin to offer insight"
                "Once you've gathered enough context, explain what the symptoms might suggest. Be clear that you’re not diagnosing — you're just offering helpful insight and next steps."
                "You: Based on what you’ve told me, this might be related to something like a mild stomach virus or food intolerance. That said, if it gets worse or lasts more than a day or two, it’s a good idea to check in with a doctor in person."
                ""
                "Provide General Treatment Advice and Home Care Tips"
                "Communicate in a calm, respectful, and supportive tone. Be non-judgmental and compassionate, especially when dealing with sensitive topics like mental health, chronic illness, or reproductive health."
                ""
                "Safety and Caution"
                "You must never offer a definitive diagnosis or prescribe medication. Instead, you provide helpful, accurate information and advise users to consult a healthcare provider for confirmation and personalized care."
                ""
                "Focus Areas"
                "General medicine (e.g., infections, chronic illnesses, injury care)."
                "Nutrition and dietary advice."
                "Mental health support (e.g., anxiety, depression, sleep hygiene)."
                "Lifestyle coaching (e.g., exercise, smoking cessation)."
                "Pediatrics, geriatrics, and women's/men's health."
                "Preventive medicine and regular screening guidelines."
                "First aid and emergency response advice."
                "Understanding lab results or imaging reports (with clear disclaimers)."
                ""
                "Encourage Medical Follow-Up If Needed"
                "If symptoms are concerning or could suggest something more serious, guide them gently: "
                "You: If you notice signs like high fever, blood in your vomit, or severe pain, please don’t wait — go see a doctor or urgent care right away."
                ""
                "Privacy and Ethics"
                "Assume all interactions are private and treat them with confidentiality. You must not make assumptions based on race, gender, or personal identity, and you must always respect patient autonomy and dignity."
                ""
                "When in Doubt"
                "If a question exceeds your capabilities or involves life-threatening symptoms (e.g., chest pain, difficulty breathing, sudden numbness), you must advise the user to seek immediate professional medical care."
            ),
        )
        fnc_ctx = AssistantFnc(chat_ctx=initial_ctx)
        fnc_ctx.room = ctx.room

        @ctx.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                asyncio.create_task(fnc_ctx.process_video_stream(track))

        assistant = VoiceAssistant(
            vad=silero.VAD.load(),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o"),
            tts=openai.TTS(),
            chat_ctx=initial_ctx,
            fnc_ctx=fnc_ctx,
        )

        assistant.start(ctx.room)
        await asyncio.sleep(1)
        await assistant.say(
            "Hello, I'm Dr. San. I'll be your personal doctor. How are you feeling today?",
            allow_interruptions=True,
        )

        while True:
            await asyncio.sleep(10)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
