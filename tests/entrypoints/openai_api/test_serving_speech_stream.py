# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI, WebSocket
from pydantic import ValidationError
from pytest_mock import MockerFixture
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vllm_omni.entrypoints.openai import serving_speech_stream as streaming_speech_module
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_speech_stream import OmniStreamingSpeechHandler
from vllm_omni.entrypoints.openai.tts_adapters.base import TTSModelAdapter
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from vllm_omni.model_executor.stage_input_processors.forced_aligner import ALIGNER_WORDS_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fake_aligner_res(pairs, words):
    """Build a stand-in for the forced-aligner stage's pooling output.

    Mirrors what rides the generator in production: ``res.outputs.data`` is a
    ``[n_words, 2]`` int32 tensor of ``[start_ms, end_ms]`` and the word strings
    travel in ``additional_information``. Decoded by ``extract_word_timestamps``.
    """
    return SimpleNamespace(
        outputs=SimpleNamespace(data=torch.tensor(pairs, dtype=torch.int32), additional_information=None),
        additional_information={ALIGNER_WORDS_KEY: list(words)},
    )


def _build_test_app(
    speech_service=None,
    *,
    idle_timeout=30.0,
    config_timeout=10.0,
    commitment_supported=False,
    mocker: MockerFixture | None = None,
):
    if speech_service is None:
        assert mocker is not None
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"RIFF" + b"\x00" * 32, "audio/wav"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-1", object(), {}))
        speech_service.forced_aligner_enabled = False

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False, collect=None):
            for chunk in (b"\x01\x02", b"\x03\x04\x05"):
                yield (chunk, 24000) if include_sample_rate else chunk

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()

    if commitment_supported:
        assert mocker is not None
        adapter = mocker.MagicMock(spec=TTSModelAdapter)
        adapter.supported_text_input_modes = frozenset({"buffered", "commitment"})
        speech_service._get_tts_adapter.return_value = adapter

    handler = OmniStreamingSpeechHandler(
        speech_service=speech_service,
        idle_timeout=idle_timeout,
        config_timeout=config_timeout,
    )
    app = FastAPI()

    @app.websocket("/v1/audio/speech/stream")
    async def ws_endpoint(websocket: WebSocket):
        await handler.handle_session(websocket)

    return app, speech_service


class TestStreamingSpeechWebSocket:
    def test_text_input_mode_defaults_to_buffered(self):
        assert streaming_speech_module.StreamingSpeechSessionConfig().text_input_mode == "buffered"

    def test_text_input_mode_accepts_commitment(self):
        assert (
            streaming_speech_module.StreamingSpeechSessionConfig(text_input_mode="commitment").text_input_mode
            == "commitment"
        )

    def test_text_input_mode_rejects_unknown_mode(self):
        with pytest.raises(ValidationError, match="text_input_mode"):
            streaming_speech_module.StreamingSpeechSessionConfig(
                text_input_mode="incremental"  # type: ignore[arg-type]
            )

    def test_adapter_text_input_mode_capabilities(self):
        assert TTSModelAdapter.supported_text_input_modes == frozenset({"buffered"})
        assert Qwen3TTSAdapter.supported_text_input_modes == frozenset({"buffered", "commitment"})

    def test_non_streaming_single_frame(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["sentence_index"] == 0
                assert start["sentence_text"] == "Hello world."
                assert start["format"] == "wav"

                audio = ws.receive_bytes()
                assert audio.startswith(b"RIFF")

                done = ws.receive_json()
                assert done == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": len(audio),
                    "error": False,
                }

                session_done = ws.receive_json()
                assert session_done == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

        assert speech_service._generate_audio_bytes.await_count == 1

    def test_commitment_mode_segments_before_eof_in_order_and_preserves_source(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(
            mocker=mocker,
            commitment_supported=True,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "language": "english",
                        "text_input_mode": "commitment",
                    }
                )
                # The unresolved numeric suffix must not leak at this packet
                # frontier. The first synthesis request is emitted only once
                # the following packet closes the sentence.
                ws.send_json({"type": "input.text", "text": "The total is 2026"})
                ws.send_json({"type": "input.text", "text": " dollars. "})

                first = ws.receive_json()
                assert first["type"] == "audio.start"
                assert first["sentence_index"] == 0
                assert first["sentence_text"] == "The total is 2026 dollars."
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"

                # This suffix has no strong boundary and is therefore flushed
                # only by EOF. Leading whitespace remains raw source text.
                ws.send_json({"type": "input.text", "text": "Thank you"})
                ws.send_json({"type": "input.done"})
                second = ws.receive_json()
                assert second["type"] == "audio.start"
                assert second["sentence_index"] == 1
                assert second["sentence_text"] == " Thank you"
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"

                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 2,
                }

        assert [call.args[0].input for call in speech_service._generate_audio_bytes.await_args_list] == [
            "The total is 2026 dollars.",
            " Thank you",
        ]
        assert all("request_id" in call.kwargs for call in speech_service._generate_audio_bytes.await_args_list)

    @pytest.mark.asyncio
    async def test_commitment_queue_backpressures_the_receiver(self, mocker: MockerFixture):
        handler = OmniStreamingSpeechHandler(speech_service=mocker.MagicMock())
        state = streaming_speech_module._CommitmentUtterance(
            index=0,
            config=streaming_speech_module.StreamingSpeechSessionConfig(),
            queue=asyncio.Queue(maxsize=1),
        )
        state.queue.put_nowait("first")
        state.segment_parts.append("second!")

        enqueue = asyncio.create_task(handler._enqueue_commitment_segment(state))
        await asyncio.sleep(0)
        assert not enqueue.done()

        assert state.queue.get_nowait() == "first"
        await asyncio.wait_for(enqueue, timeout=1)
        assert state.queue.get_nowait() == "second!"

    @pytest.mark.asyncio
    async def test_websocket_writes_are_serialized(self):
        class ProbeWebSocket:
            def __init__(self):
                self.active_writes = 0
                self.max_active_writes = 0

            async def _write(self):
                self.active_writes += 1
                self.max_active_writes = max(self.max_active_writes, self.active_writes)
                await asyncio.sleep(0)
                self.active_writes -= 1

            async def send_json(self, _data):
                await self._write()

            async def send_bytes(self, _data):
                await self._write()

        probe = ProbeWebSocket()
        websocket = streaming_speech_module._SerializedWebSocket(probe)

        await asyncio.gather(websocket.send_json({"type": "error"}), websocket.send_bytes(b"audio"))

        assert probe.max_active_writes == 1

    @pytest.mark.parametrize("language", (None, "Auto", "French"))
    def test_commitment_mode_requires_chinese_or_english(self, language, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker, commitment_supported=True)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "language": language,
                        "text_input_mode": "commitment",
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "language='Chinese' or language='English'" in error["message"]

    def test_commitment_mode_requires_model_opt_in(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)
        adapter = mocker.MagicMock(spec=TTSModelAdapter)
        adapter.supported_text_input_modes = frozenset({"buffered"})
        speech_service._get_tts_adapter.return_value = adapter

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "Chinese",
                        "text_input_mode": "commitment",
                    }
                )
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "text_input_mode='commitment' is not supported by the configured TTS model",
                }

    def test_commitment_failure_drops_queued_segments_and_finishes_at_eof(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        speech_service._generate_audio_bytes.side_effect = RuntimeError("boom")

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "First! Second! Third!"})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "boom" in error["message"]
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 0,
                    "error": True,
                }
                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 1,
                }

        assert speech_service._generate_audio_bytes.await_count == 1

    def test_commitment_utterance_limit_fails_until_eof(self, monkeypatch, mocker: MockerFixture):
        monkeypatch.setattr(streaming_speech_module, "_MAX_COMMITMENT_UTTERANCE_CHARS", 3)
        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "Chinese",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "1234"})
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Commitment utterance exceeds 3 characters",
                }
                ws.send_json({"type": "input.text", "text": "late"})
                assert "current utterance has failed" in ws.receive_json()["message"]
                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {
                    "type": "session.done",
                    "utterance_index": 0,
                    "total_sentences": 0,
                }

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_commitment_rejects_overlap_and_reconfiguration_while_draining(self, mocker: MockerFixture):
        async def slow_audio(*_args, **_kwargs):
            await asyncio.sleep(0.1)
            return b"RIFF", "audio/wav"

        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        speech_service._generate_audio_bytes.side_effect = slow_audio

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello!"})
                ws.send_json({"type": "input.done"})
                ws.send_json({"type": "input.text", "text": "overlap"})
                ws.send_json({"type": "session.config", "language": "Chinese"})

                messages = [ws.receive_json(), ws.receive_json(), ws.receive_json()]
                assert messages[0]["type"] == "audio.start"
                assert messages[1]["type"] == "error"
                assert "still active" in messages[1]["message"]
                assert messages[2]["type"] == "error"
                assert "utterance is active" in messages[2]["message"]
                assert ws.receive_bytes() == b"RIFF"
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json()["type"] == "session.done"

    def test_commitment_session_close_aborts_nonstreaming_request_once(self, mocker: MockerFixture):
        async def never_finishes(*_args, **_kwargs):
            await asyncio.sleep(3600)

        app, speech_service = _build_test_app(mocker=mocker, commitment_supported=True)
        speech_service._generate_audio_bytes.side_effect = never_finishes

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "language": "English",
                        "text_input_mode": "commitment",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello!"})
                assert ws.receive_json()["type"] == "audio.start"
                # Cancellation after EOF must still omit session.done.
                ws.send_json({"type": "input.done"})
                ws.send_json({"type": "session.close"})
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()

        speech_service.engine_client.abort.assert_awaited_once()
        request_id = speech_service.engine_client.abort.await_args.args[0]
        assert request_id.startswith("speech-stream-")

    @pytest.mark.asyncio
    async def test_commitment_cancellation_suppresses_abort_induced_terminal_events(self, mocker: MockerFixture):
        generation_started = asyncio.Event()
        abort_released_generation = asyncio.Event()

        async def abort_induced_failure(*_args, **_kwargs):
            generation_started.set()
            await abort_released_generation.wait()
            raise RuntimeError("engine request aborted")

        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(side_effect=abort_induced_failure)
        speech_service.forced_aligner_enabled = False
        handler = OmniStreamingSpeechHandler(speech_service=speech_service)
        websocket = mocker.MagicMock()
        websocket.send_json = mocker.AsyncMock()
        websocket.send_bytes = mocker.AsyncMock()
        cancellation_event = asyncio.Event()

        generation = asyncio.create_task(
            handler._generate_and_send(
                websocket,
                streaming_speech_module.StreamingSpeechSessionConfig(language="English"),
                "Hello!",
                utterance_index=0,
                sentence_index=0,
                suppress_done_on_cancel=True,
                cancellation_event=cancellation_event,
            )
        )
        await generation_started.wait()

        # Model an engine abort that wakes generation with an ordinary
        # exception instead of delivering task-level CancelledError.
        cancellation_event.set()
        abort_released_generation.set()

        assert await generation is False
        websocket.send_json.assert_awaited_once()
        assert websocket.send_json.await_args.args[0]["type"] == "audio.start"
        websocket.send_bytes.assert_not_awaited()

    def test_input_done_flushes_and_keeps_connection_open(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})

                # The same connection serves several utterances; the config sent
                # once at the top keeps applying and utterance_index rises.
                for expected_index, text in enumerate(("First utterance. ", "Second utterance. ")):
                    ws.send_json({"type": "input.text", "text": text})
                    ws.send_json({"type": "input.done"})

                    start = ws.receive_json()
                    assert start["type"] == "audio.start"
                    assert start["utterance_index"] == expected_index
                    assert start["sentence_text"] == text.strip()
                    assert ws.receive_bytes().startswith(b"RIFF")
                    assert ws.receive_json()["type"] == "audio.done"
                    assert ws.receive_json() == {
                        "type": "session.done",
                        "utterance_index": expected_index,
                        "total_sentences": 1,
                    }

        assert speech_service._generate_audio_bytes.await_count == 2
        assert [call.args[0].voice for call in speech_service._generate_audio_bytes.await_args_list] == [
            "Vivian",
            "Vivian",
        ]

    def test_sentence_index_stays_within_the_flushed_utterance(self, mocker: MockerFixture):
        # sentence_index counts within one flush and utterance_index counts the
        # flushes, so a late utterance never reports "sentence 2 of 1".
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})

                for expected_index in range(3):
                    ws.send_json({"type": "input.text", "text": "Hello world. "})
                    ws.send_json({"type": "input.done"})

                    start = ws.receive_json()
                    assert start["utterance_index"] == expected_index
                    assert start["sentence_index"] == 0
                    ws.receive_bytes()

                    done = ws.receive_json()
                    assert done["utterance_index"] == expected_index
                    assert done["sentence_index"] == 0

                    session_done = ws.receive_json()
                    assert session_done["utterance_index"] == expected_index
                    assert session_done["total_sentences"] == 1
                    assert start["sentence_index"] < session_done["total_sentences"]

    def test_session_config_between_utterances_replaces_config(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                for expected_index, voice in enumerate(("Vivian", "Serena")):
                    ws.send_json({"type": "session.config", "voice": voice})
                    ws.send_json({"type": "input.text", "text": "Hello world. "})
                    ws.send_json({"type": "input.done"})

                    assert ws.receive_json()["type"] == "audio.start"
                    ws.receive_bytes()
                    assert ws.receive_json()["type"] == "audio.done"
                    assert ws.receive_json() == {
                        "type": "session.done",
                        "utterance_index": expected_index,
                        "total_sentences": 1,
                    }

        assert [call.args[0].voice for call in speech_service._generate_audio_bytes.await_args_list] == [
            "Vivian",
            "Serena",
        ]

    def test_session_config_rejected_while_input_is_buffered(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "session.config", "voice": "Serena"})

                error = ws.receive_json()
                assert error["type"] == "error"
                assert "while input is buffered" in error["message"]

                # The buffered text survives the rejected reconfiguration.
                ws.send_json({"type": "input.done"})
                assert ws.receive_json()["sentence_text"] == "Hello world."
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

        assert speech_service._generate_audio_bytes.await_args_list[0].args[0].voice == "Vivian"

    def test_session_close_ends_connection(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

                ws.send_json({"type": "session.close"})
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_json()

    def test_idle_timeout_closes_reused_connection(self, mocker: MockerFixture):
        app, _ = _build_test_app(idle_timeout=0.05, mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                ws.receive_bytes()
                assert ws.receive_json()["type"] == "audio.done"
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

                # Holding a connection open is not free: an idle client still
                # gets timed out after the flush.
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Idle timeout: no message received",
                }

    def test_streaming_multiple_binary_frames(self, mocker: MockerFixture):
        captured_requests = []

        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False

        async def mock_prepare_speech_generation(request, request_id=None):
            captured_requests.append(request)
            return request_id or "req-stream", object(), {}

        speech_service._prepare_speech_generation = mock_prepare_speech_generation

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False):
            for chunk in (b"\x01\x02", b"\x03\x04\x05", b"\x06"):
                yield (chunk, 24000) if include_sample_rate else chunk

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "initial_codec_chunk_frames": 12,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["format"] == "pcm"
                assert start["sample_rate"] == 24000

                assert ws.receive_bytes() == b"\x01\x02"
                assert ws.receive_bytes() == b"\x03\x04\x05"
                assert ws.receive_bytes() == b"\x06"

                done = ws.receive_json()
                assert done == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 6,
                    "error": False,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

        assert len(captured_requests) == 1
        assert captured_requests[0].stream is True
        assert captured_requests[0].response_format == "pcm"
        assert captured_requests[0].initial_codec_chunk_frames == 12
        assert speech_service._generate_audio_bytes.await_count == 0

    def test_word_timestamps_requires_configured_aligner(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "word_timestamps": True,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                error = ws.receive_json()
                assert error["type"] == "error"
                assert "without --forced-aligner" in error["message"]

    def test_word_timestamps_emit_pipeline_json_frame(self, mocker: MockerFixture):
        captured_requests = []
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = True

        async def mock_prepare_speech_generation(request, request_id=None):
            captured_requests.append(request)
            return request_id or "req-stream", object(), {}

        speech_service._prepare_speech_generation = mock_prepare_speech_generation

        first_chunk = b"\x01" * 1000
        second_chunk = b"\x02" * 1000

        # The forced-aligner stage rides the same generator: its pooling output
        # is surfaced via the ``collect`` channel once the audio has streamed.
        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False, collect=None):
            for chunk in (first_chunk, second_chunk):
                yield (chunk, 1000) if include_sample_rate else chunk
            if collect is not None:
                collect["aligner_res"] = _fake_aligner_res([[0, 200], [200, 900]], ["Hello", "world"])

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "word_timestamps": True,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["type"] == "audio.start"
                assert start["word_timestamps"] is True

                # Audio streams first; timestamps are null until the sentence
                # is fully aligned.
                chunk = ws.receive_json()
                assert chunk["type"] == "audio.chunk"
                assert chunk["utterance_index"] == 0
                assert chunk["sentence_index"] == 0
                assert chunk["chunk_id"] == 0
                assert chunk["chunk_start_ms"] == 0
                assert chunk["chunk_end_ms"] == 500
                assert chunk["sample_rate"] == 1000
                assert base64.b64decode(chunk["audio_b64"]) == first_chunk
                assert chunk["timestamps"] is None

                chunk = ws.receive_json()
                assert chunk["type"] == "audio.chunk"
                assert chunk["chunk_id"] == 1
                assert chunk["chunk_start_ms"] == 500
                assert chunk["chunk_end_ms"] == 1000
                assert chunk["sample_rate"] == 1000
                assert base64.b64decode(chunk["audio_b64"]) == second_chunk
                assert chunk["timestamps"] is None

                # Final frame: empty audio carrying the whole-sentence timestamps.
                chunk = ws.receive_json()
                assert chunk["type"] == "audio.chunk"
                assert chunk["chunk_id"] == 2
                assert chunk["chunk_start_ms"] == 0
                assert chunk["chunk_end_ms"] == 1000
                assert chunk["sample_rate"] == 1000
                assert base64.b64decode(chunk["audio_b64"]) == b""
                assert chunk["timestamps"] == [
                    {"word": "Hello", "start_ms": 0, "end_ms": 200},
                    {"word": "world", "start_ms": 200, "end_ms": 900},
                ]

                done = ws.receive_json()
                assert done == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 2000,
                    "error": False,
                }

        assert captured_requests[0].word_timestamps is True

    def test_word_timestamps_emit_word_dicts(self, mocker: MockerFixture):
        # The streaming layer forwards the aligner's (already monotonic,
        # non-overlapping) words as JSON dicts in the trailing frame.
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = True
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req", object(), {}))

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False, collect=None):
            chunk = b"\x01" * 1000
            yield (chunk, 1000) if include_sample_rate else chunk
            if collect is not None:
                collect["aligner_res"] = _fake_aligner_res([[0, 1000], [1000, 1200]], ["Hello", "world"])

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                        "word_timestamps": True,
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                # Real-time audio chunk (timestamps null), then the timestamp frame.
                assert ws.receive_json()["timestamps"] is None
                final = ws.receive_json()
                assert final["type"] == "audio.chunk"
                timestamps = final["timestamps"]
                assert timestamps == [
                    {"word": "Hello", "start_ms": 0, "end_ms": 1000},
                    {"word": "world", "start_ms": 1000, "end_ms": 1200},
                ]
                for ts in timestamps:
                    assert ts["end_ms"] >= ts["start_ms"]

    def test_flush_on_input_done(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world without punctuation"})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_invalid_streaming_config(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "wav",
                    }
                )
                error = ws.receive_json()
                assert error["type"] == "error"
                assert "response_format='pcm'" in error["message"]

    def test_empty_input_text_emits_no_audio(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": ""})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_multiple_sentences_are_buffered_into_one_request(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "First sentence. "})
                ws.send_json({"type": "input.text", "text": "Second sentence. "})
                ws.send_json({"type": "input.done"})

                start = ws.receive_json()
                assert start["sentence_index"] == 0
                assert start["sentence_text"] == "First sentence. Second sentence."
                ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_unknown_message_type_keeps_session_open(self, mocker: MockerFixture):
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "unknown"})

                error = ws.receive_json()
                assert error == {"type": "error", "message": "Unknown message type: unknown"}

                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})
                assert ws.receive_json()["type"] == "audio.start"
                ws.receive_bytes()
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 36,
                    "error": False,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_config_timeout_closes_session(self, mocker: MockerFixture):
        app, _ = _build_test_app(config_timeout=0.01, mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                error = ws.receive_json()
                assert error == {"type": "error", "message": "Timeout waiting for session.config"}

    def test_generation_error_marks_audio_done(self, mocker: MockerFixture):
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(side_effect=RuntimeError("boom"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-err", object(), {}))
        speech_service._generate_pcm_chunks = mocker.AsyncMock()
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Generation failed for utterance 0, sentence 0: boom",
                }
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 0,
                    "error": True,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_streaming_generation_error_marks_audio_done(self, mocker: MockerFixture):
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-stream-err", object(), {}))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False):
            yield b"\x01\x02"
            raise RuntimeError("stream boom")

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        app, _ = _build_test_app(speech_service)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json(
                    {
                        "type": "session.config",
                        "voice": "Vivian",
                        "stream_audio": True,
                        "response_format": "pcm",
                    }
                )
                ws.send_json({"type": "input.text", "text": "Hello world. "})
                ws.send_json({"type": "input.done"})

                assert ws.receive_json()["type"] == "audio.start"
                assert ws.receive_bytes() == b"\x01\x02"
                assert ws.receive_json() == {
                    "type": "error",
                    "message": "Generation failed for utterance 0, sentence 0: stream boom",
                }
                assert ws.receive_json() == {
                    "type": "audio.done",
                    "utterance_index": 0,
                    "sentence_index": 0,
                    "total_bytes": 2,
                    "error": True,
                }

                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 1}

    def test_invalid_input_text_type_returns_validation_error(self, mocker: MockerFixture):
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": 123})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "input.text requires a string value",
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_input_text_message_too_large(self, monkeypatch, mocker: MockerFixture):
        monkeypatch.setattr(streaming_speech_module, "_MAX_INPUT_TEXT_MESSAGE_SIZE", 32)
        app, speech_service = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian"})
                ws.send_json({"type": "input.text", "text": "x" * 128})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "input.text message too large",
                }

                ws.send_json({"type": "input.done"})
                assert ws.receive_json() == {"type": "session.done", "utterance_index": 0, "total_sentences": 0}

        assert speech_service._generate_audio_bytes.await_count == 0

    def test_session_config_message_too_large(self, monkeypatch, mocker: MockerFixture):
        monkeypatch.setattr(streaming_speech_module, "_MAX_CONFIG_MESSAGE_SIZE", 64)
        app, _ = _build_test_app(mocker=mocker)

        with TestClient(app) as client:
            with client.websocket_connect("/v1/audio/speech/stream") as ws:
                ws.send_json({"type": "session.config", "voice": "Vivian", "ref_audio": "x" * 512})

                assert ws.receive_json() == {
                    "type": "error",
                    "message": "session.config message too large",
                }

    def test_disconnect_aborts_streaming_request(self, mocker: MockerFixture):
        speech_service = mocker.MagicMock(spec=OmniOpenAIServingSpeech)
        speech_service._generate_audio_bytes = mocker.AsyncMock(return_value=(b"", "audio/wav"))
        speech_service._prepare_speech_generation = mocker.AsyncMock(return_value=("req-abort", object(), {}))
        speech_service.engine_client = mocker.MagicMock()
        speech_service.engine_client.abort = mocker.AsyncMock()
        speech_service.forced_aligner_enabled = False

        async def mock_generate_pcm_chunks(_generator, _request_id, *, include_sample_rate=False):
            yield b"\x01\x02"

        speech_service._generate_pcm_chunks = mock_generate_pcm_chunks
        handler = OmniStreamingSpeechHandler(speech_service=speech_service)

        websocket = mocker.MagicMock()
        websocket.send_json = mocker.AsyncMock(side_effect=[None, WebSocketDisconnect()])
        websocket.send_bytes = mocker.AsyncMock(side_effect=WebSocketDisconnect())

        config = mocker.MagicMock()
        config.model = None
        config.voice = "Vivian"
        config.task_type = None
        config.language = None
        config.instructions = None
        config.response_format = "pcm"
        config.speed = 1.0
        config.max_new_tokens = None
        config.initial_codec_chunk_frames = None
        config.ref_audio = None
        config.ref_text = None
        config.x_vector_only_mode = None
        config.speaker_embedding = None
        config.stream_audio = True
        config.word_timestamps = False

        with pytest.raises(WebSocketDisconnect):
            asyncio.run(
                handler._generate_and_send(
                    websocket,
                    config,
                    "Hello world.",
                    utterance_index=0,
                    sentence_index=0,
                )
            )

        speech_service.engine_client.abort.assert_awaited_once_with("req-abort")
        assert websocket.send_json.await_count == 2


class TestGeneratePcmChunksContract:
    """Guard: _generate_pcm_chunks must exist on OmniOpenAIServingSpeech.

    The WebSocket handler calls speech_service._generate_pcm_chunks()
    at runtime. If the method is removed, all WS TTS streaming breaks
    with an AttributeError. This test catches that at CI time.
    """

    def test_generate_pcm_chunks_defined(self):
        assert hasattr(OmniOpenAIServingSpeech, "_generate_pcm_chunks")
        assert asyncio.iscoroutinefunction(OmniOpenAIServingSpeech._generate_pcm_chunks) or callable(
            OmniOpenAIServingSpeech._generate_pcm_chunks
        )
