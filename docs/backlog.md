# Backlog Draft

Черновик набора задач, которые потом можно перенести в GitHub Issues.

## Epic 1. Foundation

- Создать базовую структуру репозитория
- Выбрать стек POC и зафиксировать ADR
- Поднять локальную среду разработки через Docker Compose
- Настроить базовый CI
- Добавить линтеры, форматтеры и тестовый каркас

## Epic 2. Domain And Workflow

- Описать модель заявки и state machine
- Описать доменные сущности: request, reading, artifact, hardware session
- Определить контракты шагов пайплайна
- Определить retry policy и terminal states

## Epic 3. Telegram Intake

- Реализовать Telegram-бота
- Добавить прием новой заявки
- Добавить уведомления о статусе заявки
- Добавить отправку результата пользователю
- Добавить rate limiting и антиспам

## Epic 4. Validation

- Реализовать слой safety moderation
- Реализовать проверку на prompt injection
- Реализовать нормализацию пользовательского запроса
- Добавить сценарий ручной проверки сомнительных заявок

## Epic 5. Orchestration And Queue

- Поднять очередь задач
- Реализовать оркестратор шагов
- Добавить лог переходов между состояниями
- Добавить recovery после падения воркера
- Добавить retries и dead-letter сценарии

## Epic 6. LLM And Reading Generation

- Реализовать адаптер LLM
- Сформировать prompt chain для вступления
- Сформировать prompt chain для гадания по выпавшим картам
- Определить формат структурированного ответа
- Добавить тесты на стабильность выходного формата

## Epic 7. TTS And Media

- Реализовать адаптер TTS
- Определить формат аудио-артефактов
- Реализовать сборку итогового ролика
- Добавить наложение вступления и финального ответа на видео
- Добавить хранение media-артефактов

## Epic 8. Hardware Abstraction

- Описать интерфейс управления механикой
- Реализовать simulator adapter
- Реализовать журнал действий механики
- Реализовать таймауты и аварийную остановку
- Подготовить контракт для real device adapter

## Epic 9. Vision

- Собрать тестовый датасет изображений карт
- Реализовать MVP-распознавание карт
- Проверить точность на реальных условиях съемки
- Добавить fallback для ручного подтверждения карты

## Epic 10. Delivery And Operations

- Реализовать сбор итогового набора артефактов
- Добавить отправку текста, фото и видео пользователю
- Реализовать операторский интерфейс или CLI
- Добавить audit trail по каждой заявке
- Добавить метрики и алерты для POC

## Предлагаемый порядок старта

1. Foundation
2. Domain And Workflow
3. Telegram Intake
4. Validation
5. Orchestration And Queue
6. LLM And Reading Generation
7. TTS And Media
8. Hardware Abstraction
9. Vision
10. Delivery And Operations

## Минимальный vertical slice

Первый по-настоящему полезный slice:

- Telegram intake
- validation
- queue
- orchestration
- LLM text generation
- TTS
- simulator cards
- artifact delivery

Если он работает, значит проект уже живет и дальше можно заменять моки на реальные подсистемы.
