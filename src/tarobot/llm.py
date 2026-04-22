from __future__ import annotations

import re
from typing import List

from .cards import dominant_suit, major_arcana_count
from .models import DrawnCard, ReadingNarrative, ReadingRequest, SpeechPlan, SpeechSegment


SUIT_THEMES = {
    "Жезлы": "сейчас много энергии действия, амбиций и желания сдвинуть ситуацию",
    "Кубки": "главный слой вопроса эмоциональный: чувства, привязанности и интуиция",
    "Мечи": "вопрос требует ясности, честного решения и аккуратности в словах",
    "Пентакли": "ключ к ситуации лежит в практичности, деньгах, быте или ресурсе тела",
}


class MockLLMService:
    """Локальный заменитель LLM для первого MVP."""

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
        segments = [
            SpeechSegment(
                key="intro",
                section="intro",
                text=(
                    "Сейчас спокойно посмотрим на ваш вопрос. "
                    f"Вы спрашиваете: {self._normalize_question_for_speech(request.question)}. "
                    f"В раскладе сегодня {self._cards_count_phrase(len(cards))}."
                ),
                pause_ms=700,
            )
        ]

        for card in cards:
            segments.append(
                SpeechSegment(
                    key=f"card_{card.position}",
                    section="cards",
                    text=self._spoken_card_line(card),
                    pause_ms=550,
                )
            )

        segments.append(
            SpeechSegment(
                key="summary",
                section="summary",
                text=(
                    "Если собрать расклад целиком, то "
                    f"{self._lowercase_first_char(self._normalize_sentence(summary))}"
                ),
                pause_ms=700,
            )
        )
        segments.append(
            SpeechSegment(
                key="advice",
                section="outro",
                text=(
                    "И главный совет здесь такой: "
                    f"{self._lowercase_first_char(self._normalize_sentence(self._spoken_advice(advice)))}"
                ),
                pause_ms=300,
            )
        )
        return SpeechPlan(segments=segments)

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
        return question.strip().rstrip("?.!…")

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
