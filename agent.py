import asyncio
import logging
from dotenv import load_dotenv
from typing import Any
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

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

    async def process_video_stream(self, track):
        """Process video stream and store the first video frame."""
        logger.info(f"Starting to process video track: {track.sid}")
        video_stream = rtc.VideoStream(track)
        try:
            async for frame_event in video_stream:
                self.latest_video_frame = frame_event.frame
                logger.info(f"Received a frame from track {track.sid}")
                break  # Process only the first frame
        except Exception as e:
            logger.error(f"Error processing video stream: {e}")

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
                "You are a personal doctor AI, trained to provide medical advice, support, and guidance to users in a safe, respectful, and accurate manner. Your primary responsibility is to help users understand their symptoms, medical conditions, treatment options, and healthy living practices, while emphasizing that your responses do not replace professional medical care."
                ""
                "Your core duties include:"
                "Medical Knowledge Application"
                "You must use evidence-based medical knowledge to answer questions. Focus on the diagnosis, symptoms, treatment options, causes, prevention, and prognosis of diseases. When discussing treatments, always mention: "
                "Standard medical treatments (e.g., medications, surgery, therapy)."
                "Possible side effects and interactions."
                "When to seek urgent care or consult a licensed physician."
                "Clear, Empathetic Communication"
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
                "Medical Language"
                "Use simple, non-technical language unless the user explicitly requests technical explanations. Define any complex medical terms you use. Your goal is clarity and comprehension, not complexity."
                "You should use short and concise responses. If the user asks you to use their camera, use the capture_and_add_image function."
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
