# Meeting Knowledge Extractor

AI-сервис для анализа стенограмм встреч. Сервис извлекает краткую сводку, решения, задачи, ответственных, риски и базовые метрики качества результата.

## Возможности

- Web-кабинет менеджера: проекты, загрузка стенограмм, запуск анализа, фильтры задач, редактирование ответственных.
- REST API на FastAPI: `/analyze`, `/api/projects`, `/mcp/tool`, `/mcp/execute`.
- LLM-интеграция через GigaChat или DeepSeek с выбором провайдера перед запуском.
- LangGraph/LangChain pipeline для LLM-шагов: prompt -> model -> JSON parsing -> normalization.
- Агентный pipeline: контекст проекта, извлечение решений/задач/рисков и обновление статусов существующих задач.
- Память проекта: новые стенограммы сравниваются с релевантными сохраненными задачами, поэтому фразы вроде "презентация готова" закрывают найденную ранее задачу.
- Сжатая project-memory сохраняется в БД и переиспользуется в следующих запусках вместо отправки длинной истории стенограмм.
- Сроки задач сохраняются в JSON (`due_date`) для API-совместимости, но в текущем Web UI скрыты.
- Длинные стенограммы обрабатываются ParallelAgent: текст режется на чанки, чанки анализируются параллельно, затем JSON быстро объединяется локально; LLM MergeAgent можно включить отдельным флагом.
- Confidence score подсвечивает спорные задачи, неясные статусы и отсутствующих ответственных в Web UI.
- Статусы существующих задач обновляются по `task_updates`, которые возвращает выбранный LLM API.
- Если токена, сети или корректного JSON-ответа нет, сервис возвращает ошибку с причиной и не запускает локальный анализ вместо API.
- SQLite-хранилище для проектов, стенограмм, запусков, задач, решений и людей.
- Базовые unit/API/acceptance tests.

## Запуск

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

Для выбора LLM добавьте `.env`:

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

Можно оставить только один набор ключей. Без токена выбранного провайдера проект вернет ошибку с причиной.

## REST API

### Analyze

`POST /analyze`

```json
{
  "provider": "gigachat",
  "transcript": "Решили запустить пилот. Анна подготовит презентацию."
}
```

Ответ:

```json
{
  "summary": "Извлечено: решений - 1, задач - 1, рисков - 0.",
  "decisions": ["Решили запустить пилот"],
  "tasks": [
    {
      "description": "презентацию",
      "assignee": "Анна",
      "status": "todo",
      "priority": "medium",
      "due_date": "2026-06-01"
    }
  ],
  "task_updates": [
    {
      "task_id": 1,
      "description": "презентацию",
      "assignee": "Анна",
      "status": "done",
      "due_date": "2026-06-01",
      "reason": "в стенограмме сказано, что задача готова"
    }
  ],
  "people": {"Анна": ["презентацию"]},
  "risks": [],
  "agent_notes": [],
  "metrics": {
    "transcript_chars": 54,
    "llm_transcript_chars": 54,
    "decisions_count": 1,
    "tasks_count": 1,
    "people_count": 1,
    "risks_count": 0,
    "task_updates_count": 1,
    "response_time_seconds": 0.01
  },
  "source": "gigachat",
  "model_name": "GigaChat"
}
```

Если провайдер недоступен, ответ будет ошибкой:

```json
{
  "detail": {
    "provider": "deepseek",
    "reason": "DEEPSEEK_API_KEY is not configured"
  }
}
```

Ограничение по умолчанию: стенограмма до `100 000` символов. Для длинных текстов сервис хранит полный текст, но перед LLM отправляет сжатые релевантные фрагменты.

### Projects

- `GET /api/projects` - список проектов.
- `POST /api/projects` - создать проект.
- `POST /api/projects/{project_id}/transcripts` - загрузить `.txt` или текст стенограммы.
- `POST /api/projects/{project_id}/extract` - запустить извлечение для стенограммы, опционально с form-полем `provider=gigachat|deepseek`.
- `GET /api/projects/{project_id}/people` - ответственные.
- `GET /api/projects/{project_id}/tasks` - задачи.
- `DELETE /api/projects/{project_id}/tasks/{task_id}` - удалить задачу.
- `GET /api/projects/{project_id}/decisions` - решения.
- `GET /api/projects/{project_id}/runs` - история запусков.

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

Файл также можно запускать напрямую из IDE. Внутри есть безопасная настройка `sys.path`, поэтому ошибка `ModuleNotFoundError: No module named 'app'` не возникает.

## Архитектура

- `app/main.py` - FastAPI-приложение, подключение REST, MCP и Web UI.
- `app/api/routes` - REST endpoints.
- `app/web/routes.py` - серверные страницы на Jinja2.
- `app/services/llm_service.py` - sanitization, LangGraph/LangChain LLM pipeline, prompt, JSON parsing, normalization, ошибки LLM API и сжатие длинных стенограмм.
- `app/services/agents.py` - агенты памяти проекта и жизненного цикла задач; в контекст входят название проекта, описание проекта, имя TXT-файла, сжатая память проекта и только релевантные сохраненные задачи.
- `app/services/extraction_service.py` - сохранение результатов анализа в БД.
- `app/integrations/llm_api.py` - GigaChat и DeepSeek clients.
- `app/integrations/mcp_tool.py` - MCP schema, executor и CLI.
- `app/db` - SQLAlchemy-модели и SQLite engine.

## Тесты

```bash
python -m pytest
```

Покрытие:

- health check;
- `/analyze` и лимит `100 000`;
- пустая стенограмма;
- sanitization prompt injection;
- parsing JSON из LLM-ответа;
- normalization результата;
- MCP schema и execution;
- acceptance flow: проект -> стенограмма -> extraction -> задачи/люди/решения.

Тесты не ходят во внешний GigaChat: внешняя LLM заменяется mock-ответами.

## Метрики

В каждом результате возвращаются:

- `transcript_chars` - длина стенограммы;
- `llm_transcript_chars` - сколько символов ушло в LLM после сжатия;
- `decisions_count` - число решений;
- `tasks_count` - число задач;
- `people_count` - число ответственных;
- `risks_count` - число рисков;
- `task_updates_count` - число обновлений существующих задач;
- `low_confidence_count` - сколько задач, обновлений, решений и рисков требуют внимания клиента;
- `parallel_chunks_count` - сколько чанков использовал ParallelAgent для длинной стенограммы;
- `parallel_workers` - сколько параллельных LLM-запросов использовалось;
- `response_time_seconds` - время обработки.

Нефункциональное требование по времени ответа: целевой ответ до `10 секунд`; LLM-запрос зависит от выбранного провайдера и сети.

## Индивидуальные части

### Студент 1 - NLP extraction

Отвечает за `app/services/llm_service.py`: очистка стенограммы, защита от prompt injection, вызов LLM API, извлечение задач/решений/рисков и нормализация JSON.

Тесты: unit-тесты sanitization, parsing, normalization и ошибок провайдера.

Метрики: количество задач, решений, ответственных, рисков, длина стенограммы.

### Студент 2 - LLM summarization

Отвечает за GigaChat/DeepSeek prompt, структуру ответа LLM, краткое summary и обработку ошибок провайдера.

Тесты: mock LLM, невалидный JSON, отсутствие токена, проверка стабильного business response.

Метрики: response time, source/model, полнота извлечения на evaluation dataset.

### Студент 3 - API

Отвечает за FastAPI endpoints, MCP-интерфейс, проектные CRUD endpoints, загрузку стенограмм, web routes и Swagger.

Тесты: API-тесты `/health`, `/analyze`, `/mcp/tool`, `/mcp/execute`, project acceptance flow.

Метрики: HTTP status, latency, успешность project flow, валидность ограничений ввода.

## Сценарий защиты

1. Открыть `/projects` и создать проект.
2. Загрузить стенограмму до `100 000` символов.
3. Запустить LLM-анализ.
4. Показать summary, решения, задачи, ответственных, риски и историю запусков.
5. Открыть `/docs` и показать REST API.
6. Открыть `/mcp/tool` и показать MCP schema.
7. Запустить `python -m pytest` и показать успешные тесты.
