# Meeting Knowledge Extractor

AI-сервис для анализа стенограмм встреч. Приложение помогает вести проектную память: загружает стенограммы, извлекает сводку, решения, задачи, ответственных, риски, обновляет статусы существующих задач и показывает спорные AI-предложения для ручной проверки.

## Возможности

- Web-интерфейс менеджера: проекты, стенограммы, запуск LLM-анализа, задачи, ответственные, решения, история запусков.
- REST API на FastAPI: `/analyze`, `/api/projects`, `/mcp/tool`, `/mcp/execute`.
- Поддержка провайдеров LLM: GigaChat и DeepSeek, выбор через `.env`, API или форму запуска анализа.
- Проектная память: сохраненные задачи и краткая память проекта передаются в контекст новых запусков, чтобы обновлять уже найденные задачи через `task_updates`.
- Длинные стенограммы обрабатываются параллельно чанками; локальное объединение включено по умолчанию, LLM-merge включается отдельным флагом.
- Сжатие контекста перед отправкой в LLM для длинных текстов.
- Защита от простых prompt-injection фраз в стенограмме.
- Confidence score: задачи с низкой уверенностью не попадают сразу в основной список, а отображаются в очереди ручной проверки.
- Загрузка стенограмм текстом или файлом `.txt`/`.md`.
- Дата встречи определяется из стенограммы, задается вручную или по умолчанию берется как текущая дата.
- Фильтры задач по статусу, ответственному, приоритету и сортировке по дате поручения.
- Поиск внутри проекта по задачам, решениям, стенограммам и людям.
- Экспорт отчета проекта в Markdown или TXT.
- SQLite-хранилище для проектов, стенограмм, запусков, задач, AI-предложений, решений, людей и памяти проекта.
- Набор unit/API/acceptance-тестов.

## Быстрый запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

После запуска:

- Web UI: `http://127.0.0.1:8000/projects`
- Swagger: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## Docker

```bash
docker compose up --build
```

Приложение будет доступно на `http://127.0.0.1:8000`.

## Настройки

Создайте `.env` в корне проекта. Можно указать ключи только для одного провайдера, если второй не используется.

```env
LLM_PROVIDER=gigachat
GIGACHAT_TOKEN=your_token
GIGACHAT_MODEL=GigaChat

DEEPSEEK_API_KEY=your_key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_THINKING=disabled
DEEPSEEK_MAX_TOKENS=3500
DEEPSEEK_TIMEOUT_SECONDS=45
DEEPSEEK_MAX_RETRIES=1

LLM_USE_LANGGRAPH=false
MAX_TRANSCRIPT_CHARS=100000
LLM_LONG_TRANSCRIPT_CHARS=30000
LLM_TRANSCRIPT_CONTEXT_CHARS=24000
LLM_PARALLEL_LONG_TRANSCRIPTS=true
LLM_PARALLEL_LLM_MERGE=false
LLM_CHUNK_CHARS=12000
LLM_CHUNK_OVERLAP_CHARS=700
LLM_CHUNK_MAX_WORKERS=4
```

Если токен выбранного провайдера не задан или LLM вернул некорректный ответ, сервис возвращает ошибку с причиной и не подменяет результат локальным эвристическим анализом.

## Web-сценарий

1. Откройте `/projects`.
2. Создайте проект и при необходимости добавьте описание.
3. Загрузите стенограмму текстом или файлом `.txt`/`.md`; дату встречи можно указать вручную.
4. Выберите провайдера и запустите LLM-анализ.
5. Проверьте сводку, решения, риски, задачи, ответственных и историю запусков.
6. Примите или отклоните задачи с низкой уверенностью в очереди проверки.
7. Используйте фильтры, поиск и экспорт `/projects/{project_id}/export.md` или `/projects/{project_id}/export.txt`.

## REST API

### Анализ одной стенограммы

`POST /analyze`

```json
{
  "provider": "gigachat",
  "transcript": "Решили запустить пилот. Анна подготовит презентацию."
}
```

Ответ содержит бизнес-структуру:

```json
{
  "summary": "Извлечено: решений - 1, задач - 1, рисков - 0.",
  "decisions": ["Решили запустить пилот"],
  "tasks": [
    {
      "description": "подготовить презентацию",
      "assignee": "Анна",
      "status": "todo",
      "priority": "medium",
      "due_date": null
    }
  ],
  "task_updates": [],
  "people": {
    "Анна": ["подготовить презентацию"]
  },
  "risks": [],
  "metrics": {
    "transcript_chars": 54,
    "llm_transcript_chars": 54,
    "decisions_count": 1,
    "tasks_count": 1,
    "people_count": 1,
    "risks_count": 0,
    "task_updates_count": 0,
    "parallel_chunks_count": 0,
    "parallel_workers": 0,
    "response_time_seconds": 0.01
  },
  "agent_notes": [],
  "source": "gigachat",
  "model_name": "GigaChat"
}
```

Ошибка провайдера возвращается как `502`:

```json
{
  "detail": {
    "provider": "deepseek",
    "reason": "DEEPSEEK_API_KEY is not configured"
  }
}
```

Ограничение по умолчанию: стенограмма до `100000` символов.

### Проекты

- `GET /api/projects` - список проектов.
- `POST /api/projects` - создать проект.
- `GET /api/projects/{project_id}` - получить проект.
- `DELETE /api/projects/{project_id}` - удалить проект со связанными данными.
- `POST /api/projects/{project_id}/transcripts` - загрузить стенограмму текстом или файлом `.txt`/`.md`, опционально `meeting_date=YYYY-MM-DD`.
- `GET /api/projects/{project_id}/transcripts` - список стенограмм проекта.
- `POST /api/projects/{project_id}/extract` - запустить извлечение для стенограммы, form-поля `transcript_id` и опционально `provider`.
- `GET /api/projects/{project_id}/runs` - история запусков.
- `GET /api/projects/{project_id}/people` - ответственные.
- `POST /api/projects/{project_id}/people` - создать ответственного.
- `PATCH /api/projects/{project_id}/people/{person_id}` - обновить ответственного.
- `GET /api/projects/{project_id}/tasks` - задачи.
- `POST /api/projects/{project_id}/tasks` - создать задачу.
- `PATCH /api/projects/{project_id}/tasks/{task_id}` - обновить задачу.
- `DELETE /api/projects/{project_id}/tasks/{task_id}` - удалить задачу.
- `GET /api/projects/{project_id}/decisions` - решения.
- `POST /api/projects/{project_id}/decisions` - создать решение.

Статусы задач: `todo`, `in_progress`, `done`. Приоритеты: `low`, `medium`, `high`.

## MCP-инструмент

Схема инструмента:

```bash
curl http://127.0.0.1:8000/mcp/tool
```

HTTP-вызов:

```bash
curl -X POST http://127.0.0.1:8000/mcp/execute ^
  -H "Content-Type: application/json" ^
  -d "{\"transcript\":\"Анна подготовит отчет.\",\"provider\":\"deepseek\"}"
```

CLI-вызов:

```bash
python -m app.integrations.mcp_tool --transcript "Анна подготовит отчет." --provider gigachat
```

CLI также принимает JSON-строку или путь к JSON-файлу через аргументы инструмента.

## Архитектура

- `app/main.py` - FastAPI-приложение, инициализация БД, подключение REST, MCP и Web UI.
- `app/api/routes` - REST endpoints.
- `app/web/routes.py` - серверные страницы на Jinja2, фильтры, поиск, экспорт и ручная проверка AI-предложений.
- `app/services/llm_service.py` - очистка стенограммы, prompt, вызов LLM, LangGraph/LangChain pipeline, JSON parsing, normalization, обработка длинных стенограмм.
- `app/services/agents.py` - агенты проектной памяти и жизненного цикла задач.
- `app/services/extraction_service.py` - сохранение результатов анализа, задач, решений, предложений и памяти проекта в БД.
- `app/services/project_validation.py` - валидация форм, файлов, дат, статусов, приоритетов и провайдера.
- `app/integrations/llm_api.py` - клиенты GigaChat и DeepSeek.
- `app/integrations/mcp_tool.py` - MCP schema, executor и CLI.
- `app/db` - SQLAlchemy-модели и SQLite engine.
- `app/api/evaluation` - датасеты и pipeline оценки качества извлечения.

## Тесты

```bash
python -m pytest
```

Покрытие включает:

- health check;
- `/analyze`, выбор провайдера и ошибки LLM;
- лимит и валидацию стенограммы;
- очистку prompt-injection фраз;
- parsing и normalization JSON-ответа;
- DeepSeek request payload;
- параллельную обработку длинных стенограмм;
- MCP schema, HTTP execution и CLI payload loader;
- CRUD flow проекта, стенограмм, задач, людей и решений;
- определение даты встречи и привязку даты поручения к извлеченным задачам;
- удаление проекта и задач;
- загрузку `.txt`/`.md` и отказ от неподдерживаемых файлов;
- фильтры, поиск, экспорт и рендеринг Web UI;
- очередь задач с низкой уверенностью;
- deterministic scorer для evaluation pipeline.

Тесты не ходят во внешние LLM API: сетевые вызовы заменяются mock-ответами.

## Метрики результата

В ответе анализа возвращаются:

- `transcript_chars` - длина исходной стенограммы;
- `llm_transcript_chars` - сколько символов ушло в LLM после сжатия;
- `decisions_count` - число решений;
- `tasks_count` - число задач;
- `people_count` - число ответственных;
- `risks_count` - число рисков;
- `task_updates_count` - число обновлений существующих задач;
- `parallel_chunks_count` - сколько чанков использовал ParallelAgent;
- `parallel_workers` - сколько параллельных LLM-запросов использовалось;
- `response_time_seconds` - время обработки.
