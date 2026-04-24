from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tarobot.cards import draw_cards
from tarobot.llm import YandexLLMService
from tarobot.models import ReadingRequest, SpeechPlan, SpeechSegment
from tarobot.tts import YandexSpeechKitTTSProvider


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def build_test_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(22050)
        writer.writeframes(b"\x00\x00" * 400)
    return buffer.getvalue()


class YandexProviderTests(unittest.TestCase):
    def test_yandex_llm_service_parses_json_narrative(self) -> None:
        cards = draw_cards(cards_count=3, seed=42)
        captured = {}

        def fake_urlopen(request, timeout):
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse(
                {
                    "result": {
                        "alternatives": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "text": json.dumps(
                                        {
                                            "title": "Расклад на 3 карты",
                                            "summary": "Ситуация созрела для ясного решения и мягкого движения вперед.",
                                            "card_sections": [
                                                "Прошлое показывает, что раньше вы сдерживали себя сильнее, чем нужно.",
                                                "Настоящее просит честно посмотреть на реальные ограничения и ресурсы.",
                                                "Будущее намекает, что шаг будет удачным, если не разменивать энергию по мелочам.",
                                            ],
                                            "advice": "Лучше выбрать один ясный вектор и держаться его без лишней суеты.",
                                            "speech": {
                                                "intro": "Сейчас <[small]> спокойно посмотрим на ваш вопрос.",
                                                "cards": [
                                                    "Прошлое говорит: раньше вы слишком многое держали внутри.",
                                                    "Настоящее шепчет: **ясность** сейчас важнее спешки.",
                                                    "Будущее отвечает: шаг будет верным, если двигаться ровно.",
                                                ],
                                                "summary": "В целом расклад ведет вас к более собранному и взрослому решению.",
                                                "outro": "Итог простой: действуйте мягко, но не сворачивайте.",
                                            },
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                                "status": "ALTERNATIVE_STATUS_FINAL",
                            }
                        ],
                        "usage": {
                            "inputTextTokens": "100",
                            "completionTokens": "200",
                            "totalTokens": "300",
                        },
                        "modelVersion": "test-version",
                    }
                }
            )

        with patch("tarobot.llm.urlopen", side_effect=fake_urlopen):
            service = YandexLLMService(
                api_key="test-key",
                model_uri="gpt://folder/yandexgpt/latest",
                tts_voice="zamira",
                tts_role="friendly",
                tts_speed=1.0,
                tts_pitch_shift=-100,
            )
            narrative = service.generate_reading(
                ReadingRequest(question="Стоит ли идти дальше?", cards_count=3, seed=42),
                cards,
            )

        self.assertEqual(narrative.title, "Расклад на 3 карты")
        self.assertEqual(len(narrative.card_sections), 3)
        self.assertEqual(len(narrative.speech_plan.segments), 6)
        self.assertIn("sil<[400]>", captured["body"]["messages"][0]["text"])
        self.assertIn("Никогда не пиши буквальный шаблон sil<[t]>", captured["body"]["messages"][0]["text"])
        self.assertEqual(captured["body"]["modelUri"], "gpt://folder/yandexgpt/latest")
        self.assertIn("Api-Key test-key", captured["headers"].get("Authorization", ""))

    def test_yandex_tts_provider_writes_wav_file(self) -> None:
        captured = {}
        wav_bytes = build_test_wav_bytes()
        encoded = base64.b64encode(wav_bytes).decode("ascii")

        def fake_urlopen(request, timeout):
            captured.setdefault("bodies", []).append(json.loads(request.data.decode("utf-8")))
            return FakeHTTPResponse({"result": {"audioChunk": {"data": encoded}}})

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("tarobot.tts.urlopen", side_effect=fake_urlopen):
                provider = YandexSpeechKitTTSProvider(
                    api_key="test-key",
                    voice="zamira",
                    role="friendly",
                    speed=1.0,
                    pitch_shift=-100,
                )
                output_path = provider.synthesize(
                    SpeechPlan(
                        segments=[
                            SpeechSegment(
                                key="intro",
                                section="intro",
                                text="Сейчас <[small]> посмотрим на ваш вопрос.",
                                pause_ms=200,
                            )
                        ]
                    ),
                    Path(tmp_dir),
                    "reading_audio",
                )

            self.assertIsNotNone(output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.suffix, ".wav")

        body = captured["bodies"][0]
        self.assertEqual(body["hints"][0]["voice"], "zamira")
        self.assertIn({"role": "friendly"}, body["hints"])
        self.assertIn({"pitchShift": -100}, body["hints"])
        self.assertEqual(body["outputAudioSpec"]["containerAudio"]["containerAudioType"], "WAV")


if __name__ == "__main__":
    unittest.main()
