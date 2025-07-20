#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import time
from typing import Optional

import aiohttp
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from PIL import Image  # For saving debug images

# OpenGL imports
try:
    import OpenGL.GL as gl
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    logger.error("PyOpenGL not installed. Run: pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)

# GLFW import
try:
    import glfw
except ImportError:
    logger.error("glfw not installed. Run: pip install glfw")
    sys.exit(1)

# Live2D import - install with: pip install live2d-py
try:
    import live2d.v3 as live2d
except ImportError:
    logger.error("live2d-py not installed. Run: pip install live2d-py")
    sys.exit(1)

# Pipecat imports
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.examples.daily_runner import configure
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    OutputImageRawFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.ai_service import AIService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

save_to_disk = False


class SimpleLive2DRenderer:
    """
    Live2D renderer using live2d-py library.
    """

    def __init__(self, width=640, height=480, model_path=None):
        self.width = width
        self.height = height
        self.model_path = model_path
        self.window = None
        self.model = None
        self._initialized = False

    def initialize(self):
        """Initialize OpenGL context and load Live2D model"""
        if self._initialized:
            return

        # Initialize GLFW
        if not glfw.init():
            raise Exception("Failed to initialize GLFW")

        # Create hidden window (change to glfw.TRUE for debugging)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)

        # For macOS legacy OpenGL 2.1
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        self.window = glfw.create_window(self.width, self.height, "Live2D", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Failed to create window")

        glfw.make_context_current(self.window)

        live2d.init()
        live2d.glewInit()

        # Set up OpenGL
        gl.glViewport(0, 0, self.width, self.height)

        # Set pixel alignment to 1 byte to avoid padding issues
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

        # Load Live2D model
        self.model = live2d.LAppModel()
        self.model.LoadModelJson(self.model_path)
        self.model.Resize(self.width, self.height)

        self._initialized = True
        logger.info("OpenGL renderer and Live2D model initialized")

    def render_frame(self, mouth_open: float, mouth_form: float) -> np.ndarray:
        """Render a frame with Live2D model and return as numpy array"""
        if not self._initialized:
            self.initialize()

        # Make context current
        glfw.make_context_current(self.window)

        live2d.clearBuffer(0.0, 0.0, 0.0, 0.0)  # Clear to transparent

        # Update Live2D model parameters for lip-sync
        self.model.SetParameterValue("ParamMouthOpenY", mouth_open)
        self.model.SetParameterValue("ParamMouthForm", mouth_form)

        # Update and draw the model
        self.model.Update()
        self.model.Draw()

        gl.glFlush()  # Ensure rendering is complete

        # Read pixels as RGB
        pixels = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        # Convert to numpy array and flip vertically
        image = np.frombuffer(pixels, dtype=np.uint8)
        image = image.reshape(self.height, self.width, 3)
        image = np.flipud(image)

        return image

    def cleanup(self):
        """Clean up resources"""
        if self.window:
            glfw.destroy_window(self.window)
        live2d.dispose()
        glfw.terminate()
        self._initialized = False


class AudioAnalyzer:
    """Analyzes audio for lip-sync parameters"""

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate

    def analyze_for_lipsync(self, audio_data) -> dict:
        """Analyze audio and return animation parameter values"""
        if audio_data is None or len(audio_data) == 0:
            return {"mouth_open": 0.0, "mouth_form": 0.0}

        try:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)

            # Ensure 1D
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()

            # Calculate RMS for mouth opening
            rms = np.sqrt(np.mean(audio_data**2))
            mouth_open = np.clip(rms * 5.0, 0.0, 1.0)

            # Simple frequency analysis for mouth shape
            mouth_form = 0.0
            if len(audio_data) >= 512:
                fft = np.abs(np.fft.rfft(audio_data[:512]))
                freqs = np.fft.rfftfreq(512, 1 / self.sample_rate)

                # Get spectral centroid
                if np.sum(fft) > 0:
                    centroid = np.sum(freqs * fft) / np.sum(fft)
                    mouth_form = np.clip((centroid - 500) / 2000, -1.0, 1.0)

            return {
                "mouth_open": float(mouth_open),
                "mouth_form": float(mouth_form),
            }

        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {"mouth_open": 0.0, "mouth_form": 0.0}


class Live2DVideoService(AIService):
    """
    Live2D Video Service that generates avatar animations synchronized with audio.
    """

    def __init__(
        self,
        *,
        model_path: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_duration = 1.0 / fps

        # Renderer
        self._renderer = None
        self._use_opengl = True

        # Audio analyzer
        self._audio_analyzer = AudioAnalyzer()

        # Animation state
        self._mouth_open = 0.0
        self._mouth_form = 0.0
        self._target_mouth_open = 0.0
        self._target_mouth_form = 0.0
        self._last_update_time = 0.0

        # Lipsync queue for timing
        self._lipsync_queue = []  # List of (mouth_open, mouth_form, remaining_duration)

        # Rendering control
        self._render_task: Optional[asyncio.Task] = None
        self._video_queue = asyncio.Queue(maxsize=30)
        self._should_stop = False
        self._rendering_started = False
        self._video_push_task_handle = None

        # Frame counter
        self._frame_count = 0
        self._frames_pushed = 0

        logger.info(f"Live2D Video Service initialized: {model_path} ({width}x{height}@{fps}fps)")

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the service"""
        await super().setup(setup)

        # Initialize renderer here (on main thread)
        try:
            self._renderer = SimpleLive2DRenderer(self._width, self._height, self._model_path)
            self._renderer.initialize()  # Call initialize on main thread
            self._use_opengl = True
            logger.info("Using OpenGL renderer - initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenGL/Live2D: {e}.")

    async def cleanup(self):
        """Clean up resources"""
        await super().cleanup()
        self._should_stop = True
        if self._render_task and not self._render_task.done():
            await self._render_task
        if hasattr(self._renderer, "cleanup"):
            self._renderer.cleanup()

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        """Start the service"""
        await super().start(frame)

    async def stop(self, frame: EndFrame):
        """Stop the service"""
        await super().stop(frame)
        if self._video_push_task_handle:
            self._video_push_task_handle.cancel()
            try:
                await self._video_push_task_handle
            except asyncio.CancelledError:
                pass
        await self._stop_rendering()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service"""
        await super().cancel(frame)
        if self._video_push_task_handle:
            self._video_push_task_handle.cancel()
            try:
                await self._video_push_task_handle
            except asyncio.CancelledError:
                pass
        await self._stop_rendering()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames"""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            # Clear video queue on interruption
            while not self._video_queue.empty():
                await self._video_queue.get()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            # Analyze audio for lip-sync
            await self._handle_audio_frame(frame)
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def _handle_audio_frame(self, frame: TTSAudioRawFrame):
        """Handle incoming audio frame and update lip-sync parameters"""
        try:
            lipsync_params = self._audio_analyzer.analyze_for_lipsync(frame.audio)
            # Calculate chunk duration (assuming s16 mono)
            chunk_duration = len(frame.audio) / (2 * 24000)  # Cartesia default sample_rate=24000
            self._lipsync_queue.append(
                (lipsync_params["mouth_open"], lipsync_params["mouth_form"], chunk_duration)
            )
            logger.debug(
                f"Queued lipsync params: mouth_open={lipsync_params['mouth_open']:.3f}, mouth_form={lipsync_params['mouth_form']:.3f}, duration={chunk_duration:.3f}s"
            )
        except Exception as e:
            logger.error(f"Error handling audio frame: {e}")

    async def _start_rendering(self):
        """Start the rendering task"""
        if not self._rendering_started:
            self._should_stop = False
            self._render_task = asyncio.create_task(self._render_loop())
            self._rendering_started = True
            logger.info("Live2D rendering started")

    async def _stop_rendering(self):
        """Stop the rendering task"""
        if self._rendering_started:
            self._should_stop = True
            if self._render_task:
                await self._render_task
            self._rendering_started = False
            logger.info("Live2D rendering stopped")

    async def _render_loop(self):
        """Main rendering loop (async, runs on main thread)"""
        logger.info("Live2D render loop started")
        try:
            while not self._should_stop:
                frame_start = time.time()

                # Consume from lipsync queue
                if self._lipsync_queue:
                    self._target_mouth_open, self._target_mouth_form, remaining = (
                        self._lipsync_queue[0]
                    )
                    remaining -= self._frame_duration
                    if remaining <= 0:
                        self._lipsync_queue.pop(0)
                        if self._lipsync_queue:
                            self._target_mouth_open, self._target_mouth_form = self._lipsync_queue[
                                0
                            ][:2]
                        else:
                            self._target_mouth_open = 0.0
                            self._target_mouth_form = 0.0
                    else:
                        self._lipsync_queue[0] = (
                            self._target_mouth_open,
                            self._target_mouth_form,
                            remaining,
                        )
                else:
                    self._target_mouth_open = 0.0
                    self._target_mouth_form = 0.0

                # Smooth animation
                smoothing = 0.5
                self._mouth_open += (self._target_mouth_open - self._mouth_open) * smoothing
                self._mouth_form += (self._target_mouth_form - self._mouth_form) * smoothing

                # Render frame
                image_data = self._renderer.render_frame(self._mouth_open, self._mouth_form)

                # Save debug image (every 10th frame, to ./debug_frames/)
                if save_to_disk:
                    if self._frame_count % 10 == 0:
                        os.makedirs("./debug_frames", exist_ok=True)
                        img = Image.fromarray(image_data)
                        img.save(f"./debug_frames/frame_{self._frame_count:04d}.png")

                # Logging
                self._frame_count += 1
                if self._frame_count <= 3:
                    logger.info(f"Frame {self._frame_count}: rendered, shape={image_data.shape}")

                if self._frame_count % 300 == 0:
                    logger.info(f"Rendered {self._frame_count} frames")

                # DEBUG: Log mouth params every 10th frame
                if self._frame_count % 10 == 0:
                    logger.debug(
                        f"Frame {self._frame_count}: target_mouth_open={self._target_mouth_open:.3f}, smoothed_mouth_open={self._mouth_open:.3f}, queue_length={len(self._lipsync_queue)}"
                    )

                # Create output frame
                video_frame = OutputImageRawFrame(
                    image=image_data.tobytes(), size=(self._width, self._height), format="RGB"
                )

                # Queue frame (drop oldest if full)
                if self._video_queue.full():
                    await self._video_queue.get()
                await self._video_queue.put(video_frame)

                # Frame rate control
                elapsed = time.time() - frame_start
                sleep_time = max(0, self._frame_duration - elapsed)
                await asyncio.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in render loop: {e}")
            import traceback

            logger.error(traceback.format_exc())

        logger.info("Live2D render loop stopped")

    async def _video_push_task(self):
        """Consume video queue and push frames asynchronously"""
        logger.info("Video push task started")
        while not self._should_stop:
            try:
                video_frame = await self._video_queue.get()
                await self.push_frame(video_frame)
                self._frames_pushed += 1
                if self._frames_pushed % 30 == 0:
                    logger.debug(f"Pushed {self._frames_pushed} video frames")
            except Exception as e:
                logger.error(f"Error in video push task: {e}")
                await asyncio.sleep(0.1)
        logger.info("Video push task stopped")


async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Live2D Avatar Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_is_live=True,
                video_out_width=640,
                video_out_height=480,
                transcription_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        model_path = os.getenv("LIVE2D_MODEL_PATH", "./models/Haru/Haru.model3.json")
        live2d_video = Live2DVideoService(model_path=model_path, width=640, height=480, fps=30)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant with a Live2D avatar. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio and your avatar will lip-sync to match. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                context_aggregator.user(),
                llm,
                tts,
                live2d_video,
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
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])

            # Start video rendering after joining
            await live2d_video._start_rendering()
            live2d_video._video_push_task_handle = asyncio.create_task(
                live2d_video._video_push_task()
            )

            messages.append(
                {
                    "role": "system",
                    "content": "Please introduce yourself to the user as an AI assistant with a Live2D avatar.",
                }
            )
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()

        try:
            await runner.run(task)
        except KeyboardInterrupt:
            logger.info("Shutting down...")


if __name__ == "__main__":
    required_vars = ["CARTESIA_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        sys.exit(1)

    asyncio.run(main())
