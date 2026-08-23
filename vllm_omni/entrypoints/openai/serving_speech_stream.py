# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""WebSocket handler for streaming text input TTS.

In ``buffered`` mode, text is accumulated until ``input.done`` and synthesized
as one request.  In ``commitment`` mode, supported models may synthesize
irreversible sentence segments before EOF while an unresolved suffix remains
buffered by the text-commitment policy.

input.done is a flush, not a close: it ends the current utterance and the
connection stays open, so the next utterance reuses the same connection
instead of paying another WebSocket handshake. A connection ends on
session.close, on the idle timeout, or when the client closes the socket.
The session config is sticky across flushes and can be replaced by sending
another session.config between utterances.

"Utterance" here names the flush unit rather than any linguistic unit.  In
buffered mode it maps to one synthesis request; in commitment mode it may map
to several ordered synthesis requests, each ending at a policy-confirmed
boundary.

Protocol:
    Client -> Server:
        {"type": "session.config", ...}   # Session config (first message; repeatable)
        {"type": "input.text", "text": "..."} # Text chunks
        {"type": "input.done"}            # End of utterance, flush and keep connection open
        {"type": "session.close"}         # End of connection

    Server -> Client (default, word_timestamps=false):
        {"type": "audio.start", "utterance_index": 0, "sentence_index": 0,
         "sentence_text": "...", "format": "wav"}
        <binary frame: audio bytes>
        ...
        {"type": "audio.done", "utterance_index": 0, "sentence_index": 0}
        {"type": "session.done", "utterance_index": 0, "total_sentences": N}
        {"type": "error", "message": "..."}
        # session.done ends the flushed utterance, not the connection. An
        # utterance is just the flush unit: whatever text was buffered when
        # input.done arrived, of any length. utterance_index counts those
        # flushes across the connection, while sentence_index counts within
        # one of them and so pairs with total_sentences.

    Server -> Client (when word_timestamps=true):
        {"type": "audio.start", "utterance_index": 0, "sentence_index": 0,
         "sentence_text": "...", "format": "pcm"}
        {"type": "audio.chunk", "utterance_index": 0, "sentence_index": 0, "chunk_id": 0,
         "audio_b64": "<base64 PCM>", "timestamps": null}
        ...
        {"type": "audio.chunk", "audio_b64": "", "timestamps": [{"word", "start_ms", "end_ms"}, ...]}
        {"type": "audio.done", "utterance_index": 0, "sentence_index": 0}
        # Audio is JSON base64 PCM (not binary). A trailing empty-audio chunk carries the
        # full sentence-relative alignment. timestamps: list = aligned, [] = silence, null = failed.
"""

import asyncio
import base64
import json
from contextlib import aclosing
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
    StreamingSpeechSessionConfig,
)
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.model_executor.stage_input_processors.streaming_text_commitment import (
    CommitmentUpdate,
    StreamingTextCommitmentPolicy,
)
from vllm_omni.utils.forced_aligner import extract_word_timestamps

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 30.0  # seconds
_DEFAULT_CONFIG_TIMEOUT = 10.0  # seconds
_PCM_SAMPLE_RATE = 24000
_BYTES_PER_SAMPLE = 2  # 16-bit mono PCM
_MAX_CONFIG_MESSAGE_SIZE = 4 * 1024 * 1024  # allow large ref_audio payloads
_MAX_INPUT_TEXT_MESSAGE_SIZE = 128 * 1024
_MAX_COMMITMENT_UTTERANCE_CHARS = 128 * 1024
_MAX_COMMITMENT_PENDING_CHARS = 4096
_MAX_COMMITMENT_QUEUE_SIZE = 8

_UTTERANCE_EOF = object()


class _SerializedWebSocket:
    """Serialize ASGI writes made by the receiver and synthesis worker."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._send_lock = asyncio.Lock()

    async def receive_text(self) -> str:
        return await self._websocket.receive_text()

    async def send_json(self, data: object) -> None:
        async with self._send_lock:
            await self._websocket.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        async with self._send_lock:
            await self._websocket.send_bytes(data)

    async def close(self) -> None:
        async with self._send_lock:
            await self._websocket.close()


@dataclass
class _CommitmentUtterance:
    """Connection-local state for one incrementally committed utterance."""

    index: int
    config: StreamingSpeechSessionConfig
    policy: StreamingTextCommitmentPolicy = field(
        default_factory=lambda: StreamingTextCommitmentPolicy(max_pending_chars=_MAX_COMMITMENT_PENDING_CHARS)
    )
    queue: asyncio.Queue[str | object] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_MAX_COMMITMENT_QUEUE_SIZE)
    )
    segment_parts: list[str] = field(default_factory=list)
    input_chars: int = 0
    started_sentences: int = 0
    failed: bool = False
    eof: bool = False
    cancellation_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    worker: asyncio.Task[None] | None = None
    active_request_ids: set[str] = field(default_factory=set)
    aborted_request_ids: set[str] = field(default_factory=set)


class OmniStreamingSpeechHandler:
    """Handles WebSocket sessions for streaming text-input TTS.

    A connection carries one or more utterances. Buffered sessions generate
    once at input.done. Commitment sessions can generate confirmed segments
    before input.done and flush their unresolved suffix at EOF. The connection
    outlives each utterance so a client can keep synthesizing without
    reconnecting.

    Args:
        speech_service: The existing TTS serving instance (reused for
            validation and audio generation).
        idle_timeout: Max seconds to wait for a message before closing.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        speech_service: OmniOpenAIServingSpeech,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._speech_service = speech_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main loop for a single WebSocket connection.

        Serves any number of utterances. ``input.done`` flushes the current
        utterance and leaves the connection open for the next one. Rejecting a
        message is only fatal before the first valid session.config;
        afterwards the error is reported and the connection survives.
        """
        await websocket.accept()
        websocket = _SerializedWebSocket(websocket)  # type: ignore[assignment]

        config: StreamingSpeechSessionConfig | None = None
        text_parts: list[str] = []
        utterance_index = 0
        commitment: _CommitmentUtterance | None = None

        try:
            while True:
                # A commitment worker owns all audio/session events for its
                # utterance. Reap it before accepting the next utterance.
                if commitment is not None and commitment.worker is not None and commitment.worker.done():
                    await commitment.worker
                    commitment = None

                try:
                    raw, worker_finished = await self._receive_text(
                        websocket,
                        timeout=self._config_timeout if config is None else self._idle_timeout,
                        draining=commitment,
                    )
                    if worker_finished:
                        commitment = None
                        continue
                except asyncio.TimeoutError:
                    if config is None:
                        await self._send_error(websocket, "Timeout waiting for session.config")
                    else:
                        await self._send_error(websocket, "Idle timeout: no message received")
                    return

                msg = await self._parse_message(websocket, raw)
                if msg is None:
                    if config is None:
                        return  # Malformed handshake, connection closing
                    continue

                msg_type = msg.get("type")

                if msg_type == "session.config":
                    if text_parts or commitment is not None:
                        message = (
                            "session.config cannot be applied while input is buffered; send input.done first"
                            if text_parts
                            else "session.config cannot be applied while an utterance is active; "
                            "wait for session.done first"
                        )
                        await self._send_error(
                            websocket,
                            message,
                        )
                        continue
                    new_config = await self._build_config(websocket, msg)
                    if new_config is None:
                        if config is None:
                            return  # Error already sent, connection closing
                        continue  # Keep serving with the previous config
                    config = new_config

                elif config is None:
                    await self._send_error(
                        websocket,
                        f"Expected session.config, got: {msg_type}",
                    )
                    return

                elif msg_type == "input.text":
                    text = msg.get("text", "")
                    if not isinstance(text, str):
                        await self._send_error(websocket, "input.text requires a string value")
                        continue
                    if config.text_input_mode == "buffered":
                        text_parts.append(text)
                    else:
                        if commitment is not None and commitment.eof:
                            await self._send_error(
                                websocket,
                                "Previous utterance is still active; wait for session.done",
                            )
                            continue
                        if commitment is None:
                            commitment = self._start_commitment_utterance(
                                websocket,
                                config,
                                utterance_index,
                            )
                        await self._feed_commitment_text(websocket, commitment, text)

                elif msg_type == "input.done":
                    if config.text_input_mode == "commitment":
                        if commitment is not None and commitment.eof:
                            await self._send_error(
                                websocket,
                                "input.done overlaps an utterance that is still active; wait for session.done",
                            )
                            continue
                        if commitment is None:
                            await websocket.send_json(
                                {
                                    "type": "session.done",
                                    "utterance_index": utterance_index,
                                    "total_sentences": 0,
                                }
                            )
                        else:
                            await self._finish_commitment_utterance(commitment)
                        utterance_index += 1
                        continue

                    full_text = "".join(text_parts).strip()
                    text_parts.clear()
                    total_sentences = 0
                    if full_text:
                        # However long the buffered text is, the pipeline takes
                        # it as one request, so every flush is sentence 0 of 1.
                        await self._generate_and_send(
                            websocket,
                            config,
                            full_text,
                            utterance_index=utterance_index,
                            sentence_index=0,
                        )
                        total_sentences = 1

                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "utterance_index": utterance_index,
                            "total_sentences": total_sentences,
                        }
                    )
                    utterance_index += 1

                elif msg_type == "session.close":
                    await self._cleanup_commitment_utterance(commitment)
                    commitment = None
                    await websocket.close()
                    return

                else:
                    await self._send_error(
                        websocket,
                        f"Unknown message type: {msg_type}",
                    )

        except WebSocketDisconnect:
            logger.info("Streaming speech: client disconnected")
        except Exception as e:
            logger.exception("Streaming speech session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                logger.debug("Failed to send error to streaming speech client", exc_info=True)
        finally:
            await self._cleanup_commitment_utterance(commitment)

    async def _receive_text(
        self,
        websocket: WebSocket,
        *,
        timeout: float,
        draining: _CommitmentUtterance | None,
    ) -> tuple[str, bool]:
        """Receive input while noticing an EOF worker completing.

        During commitment drain the client must still be able to close the
        connection, and overlapping input must be rejected instead of being
        silently reinterpreted as the next utterance. The drain time itself is
        generation time, not client idle time; the idle clock restarts once the
        worker emits ``session.done``.
        """
        if draining is None or not draining.eof or draining.worker is None:
            return await asyncio.wait_for(websocket.receive_text(), timeout=timeout), False

        receive_task = asyncio.create_task(websocket.receive_text())
        done, _ = await asyncio.wait(
            {receive_task, draining.worker},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receive_task in done:
            return receive_task.result(), False

        receive_task.cancel()
        await asyncio.gather(receive_task, return_exceptions=True)
        await draining.worker
        return "", True

    def _start_commitment_utterance(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        utterance_index: int,
    ) -> _CommitmentUtterance:
        state = _CommitmentUtterance(index=utterance_index, config=config)
        state.worker = asyncio.create_task(self._commitment_worker(websocket, state))
        return state

    async def _feed_commitment_text(
        self,
        websocket: WebSocket,
        state: _CommitmentUtterance,
        text: str,
    ) -> None:
        if state.failed:
            await self._send_error(
                websocket,
                "input.text rejected: the current utterance has failed; send input.done",
            )
            return

        state.input_chars += len(text)
        if state.input_chars > _MAX_COMMITMENT_UTTERANCE_CHARS:
            await self._fail_commitment_input(
                websocket,
                state,
                f"Commitment utterance exceeds {_MAX_COMMITMENT_UTTERANCE_CHARS} characters",
            )
            return

        try:
            update = state.policy.feed(text)
        except ValueError as exc:
            await self._fail_commitment_input(websocket, state, str(exc))
            return
        await self._enqueue_commitment_update(state, update)

    async def _finish_commitment_utterance(self, state: _CommitmentUtterance) -> None:
        if state.eof:
            return
        state.eof = True
        if not state.failed:
            try:
                await self._enqueue_commitment_update(state, state.policy.finish())
                await self._enqueue_commitment_segment(state)
            except ValueError:
                # The pending-size failure is reported by feed(), before EOF;
                # this is defensive for a custom policy/normalizer failure.
                state.failed = True
                self._drop_queued_segments(state)
        await state.queue.put(_UTTERANCE_EOF)

    async def _enqueue_commitment_update(
        self,
        state: _CommitmentUtterance,
        update: CommitmentUpdate,
    ) -> None:
        """Segment only on boundaries already decided by the policy.

        In particular, this layer never scans punctuation again. That would
        recreate packet-seam ambiguity for decimal points and abbreviations.
        """
        for span in update.spans:
            state.segment_parts.append(span.source_text)
            if span.boundary_after:
                await self._enqueue_commitment_segment(state)

    async def _enqueue_commitment_segment(self, state: _CommitmentUtterance) -> None:
        sentence = "".join(state.segment_parts)
        state.segment_parts.clear()
        if sentence.strip() and not state.failed:
            # Awaiting a full queue deliberately backpressures receive_text.
            await state.queue.put(sentence)

    async def _fail_commitment_input(
        self,
        websocket: WebSocket,
        state: _CommitmentUtterance,
        message: str,
    ) -> None:
        if state.failed:
            return
        state.failed = True
        state.segment_parts.clear()
        self._drop_queued_segments(state)
        await self._send_error(websocket, message)

    @staticmethod
    def _drop_queued_segments(state: _CommitmentUtterance) -> None:
        while True:
            try:
                state.queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def _commitment_worker(
        self,
        websocket: WebSocket,
        state: _CommitmentUtterance,
    ) -> None:
        sentence_index = 0
        try:
            while True:
                item = await state.queue.get()
                if item is _UTTERANCE_EOF:
                    break
                if state.failed:
                    continue

                sentence_text = str(item)
                # Every count corresponds to an audio.start. Preconditions
                # which reject before audio.start therefore do not increment.
                precondition_error = self._generation_precondition_error(state.config)
                if precondition_error is not None:
                    await self._send_error(websocket, precondition_error)
                    state.failed = True
                    self._drop_queued_segments(state)
                    if state.eof:
                        break
                    continue

                state.started_sentences += 1
                succeeded = await self._generate_and_send(
                    websocket,
                    state.config,
                    sentence_text,
                    utterance_index=state.index,
                    sentence_index=sentence_index,
                    active_request_ids=state.active_request_ids,
                    aborted_request_ids=state.aborted_request_ids,
                    suppress_done_on_cancel=True,
                    cancellation_event=state.cancellation_event,
                )
                sentence_index += 1
                if not succeeded:
                    state.failed = True
                    self._drop_queued_segments(state)
                    if state.eof:
                        break
        finally:
            if state.eof and not state.cancellation_event.is_set():
                try:
                    await websocket.send_json(
                        {
                            "type": "session.done",
                            "utterance_index": state.index,
                            "total_sentences": state.started_sentences,
                        }
                    )
                except Exception:
                    logger.debug(
                        "Failed to send session.done for utterance %d",
                        state.index,
                        exc_info=True,
                    )

    async def _cleanup_commitment_utterance(
        self,
        state: _CommitmentUtterance | None,
    ) -> None:
        if state is None:
            return
        state.cancellation_event.set()
        request_ids = tuple(state.active_request_ids)
        if state.worker is not None and not state.worker.done():
            # Mark and cancel first so an abort-induced generator exception
            # cannot race out terminal events after session.close/disconnect.
            state.worker.cancel()
        for request_id in request_ids:
            await self._abort_request_once(request_id, state.aborted_request_ids)
        if state.worker is not None:
            await asyncio.gather(state.worker, return_exceptions=True)

    async def _abort_request_once(self, request_id: str, aborted: set[str]) -> None:
        if request_id in aborted:
            return
        aborted.add(request_id)
        try:
            await self._speech_service.engine_client.abort(request_id)
        except Exception:
            logger.debug("Failed to abort streaming speech request %s", request_id, exc_info=True)

    async def _parse_message(self, websocket: WebSocket, raw: str) -> dict | None:
        """Decode one client message, or report why it was rejected.

        Size limits are per message type: session.config carries ref_audio
        payloads and gets the larger budget.
        """
        if len(raw) > max(_MAX_CONFIG_MESSAGE_SIZE, _MAX_INPUT_TEXT_MESSAGE_SIZE):
            await self._send_error(websocket, "WebSocket message too large")
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
            return None

        if not isinstance(msg, dict):
            await self._send_error(websocket, "WebSocket messages must be JSON objects")
            return None

        if msg.get("type") == "session.config":
            if len(raw) > _MAX_CONFIG_MESSAGE_SIZE:
                await self._send_error(websocket, "session.config message too large")
                return None
        elif len(raw) > _MAX_INPUT_TEXT_MESSAGE_SIZE:
            await self._send_error(websocket, "input.text message too large")
            return None

        return msg

    async def _build_config(self, websocket: WebSocket, msg: dict) -> StreamingSpeechSessionConfig | None:
        """Validate a session.config message and its model."""
        try:
            config = StreamingSpeechSessionConfig(**{k: v for k, v in msg.items() if k != "type"})
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            return None

        if config.model and hasattr(self._speech_service, "_check_model"):
            error = await self._speech_service._check_model(OpenAICreateSpeechRequest(input="ping", model=config.model))
            if error is not None:
                await self._send_error(websocket, str(error))
                return None

        if config.text_input_mode == "commitment":
            adapter = self._speech_service._get_tts_adapter()
            supported_modes = adapter.supported_text_input_modes if adapter is not None else frozenset({"buffered"})
            if "commitment" not in supported_modes:
                await self._send_error(
                    websocket,
                    "text_input_mode='commitment' is not supported by the configured TTS model",
                )
                return None
            if not isinstance(config.language, str) or config.language.casefold() not in {
                "chinese",
                "english",
            }:
                await self._send_error(
                    websocket,
                    "text_input_mode='commitment' requires language='Chinese' or language='English'",
                )
                return None

        return config

    def _generation_precondition_error(self, config: StreamingSpeechSessionConfig) -> str | None:
        if not config.word_timestamps:
            return None
        if not self._speech_service.forced_aligner_enabled:
            return (
                "word_timestamps=true but the server was launched without "
                "--forced-aligner; either restart the server with that flag "
                "or set word_timestamps=false in session.config."
            )
        response_format = config.response_format or "wav"
        if not (config.stream_audio and response_format == "pcm"):
            return (
                "word_timestamps=true requires stream_audio=true and "
                "response_format='pcm' (timestamps ride the per-sentence "
                "PCM audio.chunk stream)."
            )
        return None

    async def _generate_and_send(
        self,
        websocket: WebSocket,
        config: StreamingSpeechSessionConfig,
        sentence_text: str,
        *,
        utterance_index: int,
        sentence_index: int,
        active_request_ids: set[str] | None = None,
        aborted_request_ids: set[str] | None = None,
        suppress_done_on_cancel: bool = False,
        cancellation_event: asyncio.Event | None = None,
    ) -> bool:
        """Generate audio for a single sentence and send it over WebSocket.

        ``utterance_index`` identifies the flush this sentence belongs to and
        ``sentence_index`` its position inside that flush.
        """
        response_format = config.response_format or "wav"

        # Reject unmet word-timestamps preconditions early with a clear reason.
        precondition_error = self._generation_precondition_error(config)
        if precondition_error is not None:
            await self._send_error(websocket, precondition_error)
            return False

        request = OpenAICreateSpeechRequest(
            input=sentence_text,
            model=config.model,
            voice=config.voice,
            task_type=config.task_type,
            language=config.language,
            instructions=config.instructions,
            response_format=response_format,
            speed=config.speed,
            max_new_tokens=config.max_new_tokens,
            initial_codec_chunk_frames=config.initial_codec_chunk_frames,
            non_streaming_mode=config.non_streaming_mode,
            ref_audio=config.ref_audio,
            ref_text=config.ref_text,
            x_vector_only_mode=config.x_vector_only_mode,
            speaker_embedding=config.speaker_embedding,
            stream=config.stream_audio,
            word_timestamps=config.word_timestamps,
        )

        start_payload = {
            "type": "audio.start",
            "utterance_index": utterance_index,
            "sentence_index": sentence_index,
            "sentence_text": sentence_text,
            "format": response_format,
        }
        if config.stream_audio and response_format == "pcm":
            # Nominal stream rate; each audio.chunk carries the authoritative
            # per-chunk sample_rate.
            start_payload["sample_rate"] = _PCM_SAMPLE_RATE
        if config.word_timestamps:
            start_payload["word_timestamps"] = True
        await websocket.send_json(start_payload)

        total_bytes = 0
        generation_failed = False
        generation_cancelled = False
        request_id = f"speech-stream-{random_uuid()}"
        if active_request_ids is not None:
            active_request_ids.add(request_id)
        try:
            if config.stream_audio:
                prepared_request_id, generator, _ = await self._speech_service._prepare_speech_generation(
                    request,
                    request_id=request_id,
                )
                if prepared_request_id != request_id:
                    if active_request_ids is not None:
                        active_request_ids.discard(request_id)
                        active_request_ids.add(prepared_request_id)
                    request_id = prepared_request_id
                if config.word_timestamps:
                    total_bytes = await self._stream_audio_with_alignments(
                        websocket=websocket,
                        request_id=request_id,
                        generator=generator,
                        sentence_text=sentence_text,
                        utterance_index=utterance_index,
                        sentence_index=sentence_index,
                        language=config.language,
                    )
                else:
                    async with aclosing(self._speech_service._generate_pcm_chunks(generator, request_id)) as stream:
                        async for chunk in stream:
                            total_bytes += len(chunk)
                            await websocket.send_bytes(chunk)
            else:
                audio_bytes, _ = await self._speech_service._generate_audio_bytes(
                    request,
                    request_id=request_id,
                )
                total_bytes = len(audio_bytes)
                await websocket.send_bytes(audio_bytes)
        except WebSocketDisconnect:
            if request_id is not None:
                await self._abort_request_once(
                    request_id,
                    aborted_request_ids if aborted_request_ids is not None else set(),
                )
            raise
        except asyncio.CancelledError:
            generation_cancelled = True
            raise
        except Exception as e:
            if cancellation_event is not None and cancellation_event.is_set():
                generation_cancelled = True
            else:
                generation_failed = True
                logger.error(
                    "Generation failed for utterance %d, sentence %d: %s",
                    utterance_index,
                    sentence_index,
                    e,
                )
                await self._send_error(
                    websocket,
                    f"Generation failed for utterance {utterance_index}, sentence {sentence_index}: {e}",
                )
        finally:
            if request_id is not None and active_request_ids is not None:
                active_request_ids.discard(request_id)
            try:
                cancellation_requested = cancellation_event is not None and cancellation_event.is_set()
                if not (suppress_done_on_cancel and (generation_cancelled or cancellation_requested)):
                    await websocket.send_json(
                        {
                            "type": "audio.done",
                            "utterance_index": utterance_index,
                            "sentence_index": sentence_index,
                            "total_bytes": total_bytes,
                            "error": generation_failed,
                        }
                    )
            except Exception:
                logger.debug("Failed to send audio.done for sentence %d", sentence_index, exc_info=True)
        return not generation_failed and not generation_cancelled

    async def _stream_audio_with_alignments(
        self,
        *,
        websocket: WebSocket,
        request_id: str,
        generator,
        sentence_text: str,
        utterance_index: int,
        sentence_index: int,
        language: str | None = None,
    ) -> int:
        """Stream PCM as JSON ``audio.chunk`` frames, aligned per sentence.

        Forward each PCM chunk live (``timestamps: null``). The forced-aligner
        pipeline stage (appended when the server is launched with
        ``--forced-aligner``) consumes the synthesized audio internally and its
        pooling output rides the same generator, so once the audio finishes we
        pull the word timestamps straight off that aligner output and emit a
        final empty-audio ``audio.chunk`` carrying them. Timestamps is ``null``
        when the aligner produced none; audio always flows regardless.
        """
        audio_bytes_seen = 0
        total_bytes = 0
        sample_rate = _PCM_SAMPLE_RATE
        chunk_id = 0
        # Receives the aligner stage's pooling output from the generator.
        collect: dict = {}

        async def send_chunk(
            chunk: bytes,
            chunk_sample_rate: int,
            timestamps_payload: list[dict] | None,
            chunk_start_ms: int,
            chunk_end_ms: int,
        ) -> None:
            nonlocal chunk_id
            await websocket.send_json(
                {
                    "type": "audio.chunk",
                    "utterance_index": utterance_index,
                    "sentence_index": sentence_index,
                    "chunk_id": chunk_id,
                    "chunk_start_ms": chunk_start_ms,
                    "chunk_end_ms": chunk_end_ms,
                    "sample_rate": chunk_sample_rate,
                    "audio_b64": base64.b64encode(chunk).decode("ascii"),
                    "timestamps": timestamps_payload,
                }
            )
            chunk_id += 1

        async with aclosing(
            self._speech_service._generate_pcm_chunks(
                generator,
                request_id,
                include_sample_rate=True,
                collect=collect,
            )
        ) as stream:
            async for chunk, chunk_sample_rate in stream:
                sample_rate = chunk_sample_rate
                chunk_start_ms = int(round((audio_bytes_seen / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                audio_bytes_seen += len(chunk)
                chunk_end_ms = int(round((audio_bytes_seen / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
                total_bytes += len(chunk)
                # Audio first, timestamps after the whole sentence is aligned.
                await send_chunk(chunk, chunk_sample_rate, None, chunk_start_ms, chunk_end_ms)

        # Pull word timestamps off the aligner stage's pooling output (it rode
        # the same generator); extract_word_timestamps re-segments the sentence
        # text for the word strings when the aligner output doesn't carry them.
        aligner_res = collect.get("aligner_res")
        timestamps_payload = (
            extract_word_timestamps(aligner_res, sentence_text, language) if aligner_res is not None else None
        )
        sentence_end_ms = int(round((audio_bytes_seen / _BYTES_PER_SAMPLE / sample_rate) * 1000.0))
        await send_chunk(b"", sample_rate, timestamps_payload, 0, sentence_end_ms)

        return total_bytes

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send an error message to the client."""
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": message,
                }
            )
        except Exception:
            pass  # Connection may already be closed; safe to ignore
