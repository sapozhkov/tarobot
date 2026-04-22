from __future__ import annotations

from typing import List

from .cards import dominant_suit, major_arcana_count
from .models import DrawnCard, ReadingNarrative, ReadingRequest


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
        spoken_text = self._spoken_text(request, cards, summary, advice)

        return ReadingNarrative(
            title=f"Расклад на {len(cards)} карт",
            summary=f"{lead} {summary}",
            card_sections=sections,
            advice=advice,
            spoken_text=spoken_text,
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

    def _spoken_text(
        self,
        request: ReadingRequest,
        cards: List[DrawnCard],
        summary: str,
        advice: str,
    ) -> str:
        opening = (
            f"Смотрю на ваш вопрос: {request.question}. "
            f"В раскладе выпали {len(cards)} карт."
        )

        card_lines = [
            f"{card.position_label}: {card.card.name}. {card.meaning}."
            for card in cards
        ]

        spoken = " ".join([opening, *card_lines, summary, advice])
        return spoken[:1800]
