from __future__ import annotations

import json
import re
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .cards import dominant_suit, major_arcana_count
from .models import DrawnCard, ReadingNarrative, ReadingRequest, SpeechPlan, SpeechSegment
from .prompts import build_yandex_reading_system_prompt, build_yandex_reading_user_prompt


SUIT_THEMES = {
    "Жезлы": "сейчас много энергии действия, амбиций и желания сдвинуть ситуацию",
    "Кубки": "главный слой вопроса эмоциональный: чувства, привязанности и интуиция",
    "Мечи": "вопрос требует ясности, честного решения и аккуратности в словах",
    "Пентакли": "ключ к ситуации лежит в практичности, деньгах, быте или ресурсе тела",
}


class ReadingGenerator:
    backend_name = "unknown"

    def generate_reading(self, request: ReadingRequest, cards: List[DrawnCard]) -> ReadingNarrative:
        raise NotImplementedError

    def metadata(self) -> Dict[str, str]:
        return {}


class MockLLMService(ReadingGenerator):
    """Локальный заменитель LLM для первого MVP."""

    backend_name = "mock"

    def generate_reading(self, request: ReadingRequest, cards: List[DrawnCard]) -> ReadingNarrative:
        lead = self._lead(cards)
        sections = [self._section_for_card(request, card) for card in cards]
        summary = self._summary(cards)
        advice = self._advice(cards)
        speech_plan = self._speech_plan(request, cards, summary, advice)

        return ReadingNarrative(
            title=self._spread_title(len(cards)),
            summary=f"{lead} {summary}",
            card_sections=sections,
            advice=advice,
            speech_plan=speech_plan,
        )

    def _lead(self, cards: List[DrawnCard]) -> str:
        majors = major_arcana_count(cards)
        if majors >= max(2, len(cards) // 2):
            return "В раскладе много старших арканов, поэтому ситуация ощущается не случайной и поворотной."
        return "Расклад показывает скорее рабочую, жизненную динамику, где многое зависит от ваших текущих решений."

    def _section_for_card(self, request: ReadingRequest, card: DrawnCard) -> str:
        orientation = "в прямом положении" if card.orientation == "upright" else "в перевернутом положении"
        return (
            f"{card.position_label}: {card.card.name} {orientation}. "
            f"В контексте вопроса «{request.question}» карта указывает на {card.meaning}."
        )

    def _summary(self, cards: List[DrawnCard]) -> str:
        suit = dominant_suit(cards)
        suit_summary = SUIT_THEMES.get(suit, "Здесь сильнее всего ощущается общий фон перемен и переосмысления")
        reversed_count = sum(1 for card in cards if card.orientation == "reversed")

        if reversed_count >= max(2, len(cards) // 2):
            balance = "При этом часть энергии сейчас идет с задержкой, через внутренние узлы, сомнения или сопротивление."
        else:
            balance = "Движение по вопросу возможно без резкого надрыва, если держать темп и не суетиться."

        return f"{suit_summary.capitalize()}. {balance}"

    def _advice(self, cards: List[DrawnCard]) -> str:
        suit = dominant_suit(cards)
        if suit == "Жезлы":
            return "Совет расклада: действовать смело, но не расплескать импульс на десять направлений сразу."
        if suit == "Кубки":
            return "Совет расклада: сначала назвать свои настоящие чувства, а уже потом принимать решение."
        if suit == "Мечи":
            return "Совет расклада: навести ясность в формулировках, ожиданиях и границах."
        if suit == "Пентакли":
            return "Совет расклада: опереться на конкретику, режим, деньги и проверяемые шаги."
        return "Совет расклада: не торопить итог, а собрать больше ясности и шаг за шагом выровнять ситуацию."

    def _speech_plan(
        self,
        request: ReadingRequest,
        cards: List[DrawnCard],
        summary: str,
        advice: str,
    ) -> SpeechPlan:
        return build_speech_plan(
            intro=(
                "Сейчас спокойно посмотрим на ваш вопрос. "
                f"Вы спрашиваете: {self._normalize_question_for_speech(request.question)}. "
                f"В раскладе сегодня {self._cards_count_phrase(len(cards))}."
            ),
            cards=[self._spoken_card_line(card) for card in cards],
            summary=(
                "Если собрать расклад целиком, то "
                f"{self._lowercase_first_char(self._normalize_sentence(summary))}"
            ),
            outro=(
                "И главный совет здесь такой: "
                f"{self._lowercase_first_char(self._normalize_sentence(self._spoken_advice(advice)))}"
            ),
        )

    def _spoken_card_line(self, card: DrawnCard) -> str:
        orientation = "в прямом положении" if card.orientation == "upright" else "в перевернутом положении"
        meaning = self._normalize_meaning_for_speech(card.meaning)
        return (
            f"{card.position_label}. {card.card.name}, {orientation}. "
            f"Эта карта говорит о том, что {meaning}"
        )

    def _spoken_advice(self, advice: str) -> str:
        prefix = "Совет расклада:"
        if advice.startswith(prefix):
            return advice[len(prefix):].strip()
        return advice

    def _normalize_question_for_speech(self, question: str) -> str:
        normalized = question.strip().rstrip("?.!…")
        return normalized.replace("Tarobot", "Таробот").replace("tarobot", "Таробот")

    def _cards_count_phrase(self, cards_count: int) -> str:
        if cards_count % 10 == 1 and cards_count % 100 != 11:
            return f"{cards_count} карта"
        if cards_count % 10 in {2, 3, 4} and cards_count % 100 not in {12, 13, 14}:
            return f"{cards_count} карты"
        return f"{cards_count} карт"

    def _spread_title(self, cards_count: int) -> str:
        if cards_count % 10 == 1 and cards_count % 100 != 11:
            return f"Расклад на {cards_count} карту"
        if cards_count % 10 in {2, 3, 4} and cards_count % 100 not in {12, 13, 14}:
            return f"Расклад на {cards_count} карты"
        return f"Расклад на {cards_count} карт"

    def _normalize_meaning_for_speech(self, meaning: str) -> str:
        normalized = meaning.replace("; тема карты:", ". Отдельно здесь звучит тема:")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = normalized.rstrip(".")
        if normalized:
            normalized = normalized[0].lower() + normalized[1:]
        return f"{normalized}."

    def _normalize_sentence(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized.endswith((".", "!", "?", "…")):
            normalized += "."
        return normalized[:600]

    def _lowercase_first_char(self, text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:]


class YandexLLMService(ReadingGenerator):
    backend_name = "yandex-completion"
    endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def __init__(
        self,
        *,
        api_key: str,
        model_uri: str,
        temperature: float = 0.75,
        max_tokens: int = 1800,
        timeout_seconds: int = 60,
        tts_voice: str = "marina",
        tts_role: Optional[str] = None,
        tts_speed: float = 1.0,
        tts_pitch_shift: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model_uri = model_uri
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.tts_voice = tts_voice
        self.tts_role = tts_role
        self.tts_speed = tts_speed
        self.tts_pitch_shift = tts_pitch_shift
        self._last_metadata: Dict[str, str] = {"llm_model_uri": model_uri}

    def metadata(self) -> Dict[str, str]:
        return dict(self._last_metadata)

    def generate_reading(self, request: ReadingRequest, cards: List[DrawnCard]) -> ReadingNarrative:
        payload = {
            "modelUri": self.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens),
                "reasoningOptions": {"mode": "DISABLED"},
            },
            "messages": [
                {
                    "role": "system",
                    "text": build_yandex_reading_system_prompt(
                        voice=self.tts_voice,
                        role=self.tts_role,
                        speed=self.tts_speed,
                        pitch_shift=self.tts_pitch_shift,
                    ),
                },
                {
                    "role": "user",
                    "text": build_yandex_reading_user_prompt(request, cards),
                },
            ],
        }

        response = self._post_json(payload)
        result = response.get("result", response)
        alternatives = result.get("alternatives") or []
        if not alternatives:
            raise RuntimeError("Yandex LLM не вернул ни одной альтернативы ответа")

        alternative = alternatives[0]
        status = alternative.get("status", "UNKNOWN")
        if status == "ALTERNATIVE_STATUS_CONTENT_FILTER":
            raise RuntimeError("Yandex LLM отклонил ответ из-за content filter")

        message = alternative.get("message", {})
        raw_text = str(message.get("text", "")).strip()
        if not raw_text:
            raise RuntimeError("Yandex LLM вернул пустой текстовый ответ")

        parsed = self._parse_json_response(raw_text)
        narrative = self._build_narrative(parsed, cards)

        usage = result.get("usage", {})
        self._last_metadata = {
            "llm_model_uri": self.model_uri,
            "llm_model_version": str(result.get("modelVersion", "")),
            "llm_status": str(status),
            "llm_input_tokens": str(usage.get("inputTextTokens", "")),
            "llm_completion_tokens": str(usage.get("completionTokens", "")),
            "llm_total_tokens": str(usage.get("totalTokens", "")),
        }
        return narrative

    def _post_json(self, payload: Dict[str, object]) -> Dict[str, object]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Запрос к Yandex LLM завершился ошибкой HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Не удалось выполнить запрос к Yandex LLM: {exc.reason}") from exc

    def _parse_json_response(self, text: str) -> Dict[str, object]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed

        raise RuntimeError("Yandex LLM не вернул корректный JSON-объект")

    def _build_narrative(self, payload: Dict[str, object], cards: List[DrawnCard]) -> ReadingNarrative:
        title = str(payload.get("title", "")).strip() or MockLLMService()._spread_title(len(cards))
        summary = self._require_text(payload, "summary")
        advice = self._require_text(payload, "advice")
        card_sections = self._require_text_list(payload, "card_sections", len(cards))

        speech = payload.get("speech")
        if not isinstance(speech, dict):
            raise RuntimeError("Ответ Yandex LLM должен содержать объект speech")

        intro = self._require_text(speech, "intro")
        speech_cards = self._require_text_list(speech, "cards", len(cards))
        speech_summary = self._require_text(speech, "summary")
        speech_outro = self._require_text(speech, "outro")

        return ReadingNarrative(
            title=title,
            summary=summary,
            card_sections=card_sections,
            advice=advice,
            speech_plan=build_speech_plan(
                intro=intro,
                cards=speech_cards,
                summary=speech_summary,
                outro=speech_outro,
            ),
        )

    def _require_text(self, payload: Dict[str, object], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Поле '{key}' в ответе Yandex LLM должно быть непустой строкой")
        return value.strip()

    def _require_text_list(self, payload: Dict[str, object], key: str, expected_length: int) -> List[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise RuntimeError(f"Поле '{key}' в ответе Yandex LLM должно быть списком")
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if len(normalized) != expected_length:
            raise RuntimeError(
                f"Поле '{key}' в ответе Yandex LLM должно содержать ровно {expected_length} элементов"
            )
        return normalized


def build_speech_plan(
    *,
    intro: str,
    cards: List[str],
    summary: str,
    outro: str,
) -> SpeechPlan:
    segments = [SpeechSegment(key="intro", section="intro", text=intro.strip(), pause_ms=700)]
    segments.extend(
        SpeechSegment(key=f"card_{index}", section="cards", text=text.strip(), pause_ms=550)
        for index, text in enumerate(cards, start=1)
    )
    segments.append(SpeechSegment(key="summary", section="summary", text=summary.strip(), pause_ms=700))
    segments.append(SpeechSegment(key="advice", section="outro", text=outro.strip(), pause_ms=300))
    return SpeechPlan(segments=segments)
