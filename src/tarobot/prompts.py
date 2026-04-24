from __future__ import annotations

from typing import Iterable, List, Optional

from .models import DrawnCard, ReadingRequest


def build_yandex_reading_system_prompt(
    *,
    voice: str,
    role: Optional[str],
    speed: float,
    pitch_shift: float,
) -> str:
    role_hint = role or "neutral"
    tone_hint = _tone_hint(role_hint, pitch_shift)

    return f"""
Ты пишешь текст гадания для проекта Tarobot.
Отвечай только по-русски.
Верни строго один JSON-объект без markdown-блоков, без пояснений до и после JSON.

Требуемый JSON:
{{
  "title": "краткий заголовок расклада",
  "summary": "цельный абзац с общей трактовкой",
  "card_sections": ["абзац по карте 1", "абзац по карте 2"],
  "advice": "короткий практический вывод",
  "speech": {{
    "intro": "короткое вступление для озвучки",
    "cards": ["короткая озвучка карты 1", "короткая озвучка карты 2"],
    "summary": "короткая сводка для озвучки",
    "outro": "короткий финальный вывод для озвучки"
  }}
}}

Правила для обычного текста:
- title, summary, card_sections и advice должны звучать атмосферно, но без театрального перебора.
- card_sections и speech.cards должны идти в том же порядке, что и карты во входных данных.
- card_sections и speech.cards должны содержать ровно столько элементов, сколько карт в раскладе.
- Не выдумывай новые карты и не меняй их положения.
- Не используй списки, markdown-заголовки, emoji и служебные комментарии.

Правила для блока speech:
- speech предназначен для Yandex SpeechKit TTS markup.
- Используй только TTS markup, не SSML.
- Разрешенные средства разметки:
  * контекстные паузы: <[tiny]>, <[small]>, <[medium]>, <[large]>, <[huge]>
  * явные паузы: sil<[400]>, sil<[700]> или другое число миллисекунд
  * акцент: **слово**
  * ударение: + перед гласной
  * фонемы: [[...]] только если без них слово почти наверняка произнесется плохо
- Используй разметку умеренно: не больше одной явной паузы и не больше одного акцента на сегмент, если без этого можно обойтись.
- Никогда не пиши буквальный шаблон sil<[t]>; вместо t всегда должно быть число.
- Каждый speech-сегмент должен быть коротким: 1-2 предложения, желательно до 220 символов.
- Не пиши скобки, подпункты, кавычки-елочки и лишние двоеточия, если они не нужны для звучания.
- Текст должен хорошо звучать вслух: короткие фразы, ясный ритм, без канцелярита.

Профиль голоса:
- voice: {voice}
- role: {role_hint}
- speed: {speed}
- pitch_shift: {pitch_shift}
- Желаемая подача: {tone_hint}
""".strip()


def build_yandex_reading_user_prompt(request: ReadingRequest, cards: Iterable[DrawnCard]) -> str:
    lines: List[str] = [
        f"Вопрос пользователя: {request.question}",
        f"Количество карт: {request.cards_count}",
        "Карты в раскладе:",
    ]

    for card in cards:
        orientation = "прямое положение" if card.orientation == "upright" else "перевернутое положение"
        lines.append(
            (
                f"{card.position}. {card.position_label} — {card.card.name}, {orientation}. "
                f"Базовое значение: {card.meaning}."
            )
        )

    lines.extend(
        [
            "",
            "Задача:",
            "- Сначала дай цельное толкование расклада.",
            "- Затем дай отдельные card_sections по каждой карте.",
            "- Потом дай advice.",
            "- В блоке speech сохрани ту же смысловую структуру, но сделай ее короче и пригодной для TTS.",
            "- Не превращай гадание в сухую инструкцию; нужен теплый, уверенный, немного мистический тон.",
        ]
    )
    return "\n".join(lines)


def _tone_hint(role: str, pitch_shift: float) -> str:
    lowered = pitch_shift < -20
    raised = pitch_shift > 20

    if role == "friendly":
        base = "теплая, мягкая, доверительная, без суеты"
    elif role == "strict":
        base = "собранная, ясная, спокойная, с внутренней опорой"
    elif role == "whisper":
        base = "очень интимная, тихая, почти шепотом, но без хоррора"
    else:
        base = "спокойная, естественная, живая, без механического тона"

    if lowered:
        return f"{base}; голос ощущается чуть ниже обычного"
    if raised:
        return f"{base}; голос ощущается чуть выше обычного"
    return base
