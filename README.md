# Meeting Knowledge Extractor

AI-сервис для анализа стенограмм встреч. Сервис извлекает краткую сводку, решения, задачи, ответственных, риски и базовые метрики качества результата.

## Возможности

- Web-кабинет менеджера: проекты, загрузка стенограмм, запуск анализа, фильтры задач, редактирование ответственных.
- REST API на FastAPI: `/analyze`, `/api/projects`, `/mcp/tool`, `/mcp/execute`.
- LLM-интеграция через GigaChat.
- Fallback extractor: если токена или сети нет, сервис использует локальные правила и не ломает демо.
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

Для GigaChat добавьте `.env`:

```env
GIGACHAT_TOKEN=your_token
GIGACHAT_MODEL=GigaChat
```

Без токена проект продолжит работать через fallback extractor.

## REST API

### Analyze

`POST /analyze`

```json
{
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
      "due_date": null
    }
  ],
  "people": {"Анна": ["презентацию"]},
  "risks": [],
  "metrics": {
    "transcript_chars": 54,
    "decisions_count": 1,
    "tasks_count": 1,
    "people_count": 1,
    "risks_count": 0,
    "response_time_seconds": 0.01
  },
  "source": "fallback",
  "model_name": "rule-based"
}
```

Ограничение: стенограмма до `20 000` символов.

### Projects

- `GET /api/projects` - список проектов.
- `POST /api/projects` - создать проект.
- `POST /api/projects/{project_id}/transcripts` - загрузить `.txt` или текст стенограммы.
- `POST /api/projects/{project_id}/extract` - запустить извлечение для стенограммы.
- `GET /api/projects/{project_id}/people` - ответственные.
- `GET /api/projects/{project_id}/tasks` - задачи.
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
  -d "{\"transcript\":\"Анна подготовит отчет.\"}"
```

CLI-вызов:

```bash
python -m app.integrations.mcp_tool --transcript "Анна подготовит отчет."
```

Файл также можно запускать напрямую из IDE. Внутри есть безопасная настройка `sys.path`, поэтому ошибка `ModuleNotFoundError: No module named 'app'` не возникает.

## Архитектура

- `app/main.py` - FastAPI-приложение, подключение REST, MCP и Web UI.
- `app/api/routes` - REST endpoints.
- `app/web/routes.py` - серверные страницы на Jinja2.
- `app/services/llm_service.py` - sanitization, prompt, JSON parsing, normalization, fallback extraction.
- `app/services/extraction_service.py` - сохранение результатов анализа в БД.
- `app/integrations/llm_api.py` - GigaChat client.
- `app/integrations/mcp_tool.py` - MCP schema, executor и CLI.
- `app/db` - SQLAlchemy-модели и SQLite engine.

## Тесты

```bash
python -m pytest
```

Покрытие:

- health check;
- `/analyze` и лимит `20 000`;
- пустая стенограмма;
- sanitization prompt injection;
- parsing JSON из LLM-ответа;
- normalization результата;
- MCP schema и execution;
- acceptance flow: проект -> стенограмма -> extraction -> задачи/люди/решения.

Тесты не ходят во внешний GigaChat: внешняя LLM заменяется mock/fallback.

## Метрики

В каждом результате возвращаются:

- `transcript_chars` - длина стенограммы;
- `decisions_count` - число решений;
- `tasks_count` - число задач;
- `people_count` - число ответственных;
- `risks_count` - число рисков;
- `response_time_seconds` - время обработки.

Нефункциональное требование по времени ответа: целевой ответ до `10 секунд`. Fallback обычно отвечает быстрее секунды; LLM-запрос зависит от GigaChat и сети.

## Индивидуальные части

### Студент 1 - NLP extraction

Отвечает за `app/services/llm_service.py`: очистка стенограммы, защита от prompt injection, rule-based fallback, извлечение задач/решений/рисков и нормализация JSON.

Тесты: unit-тесты sanitization, parsing, normalization и fallback extraction.

Метрики: количество задач, решений, ответственных, рисков, длина стенограммы.

### Студент 2 - LLM summarization

Отвечает за GigaChat prompt, структуру ответа LLM, краткое summary, обработку ошибок провайдера и fallback при сбоях.

Тесты: mock LLM, невалидный JSON, отсутствие токена, проверка стабильного business response.

Метрики: response time, source/model, полнота извлечения на evaluation dataset.

### Студент 3 - API

Отвечает за FastAPI endpoints, MCP-интерфейс, проектные CRUD endpoints, загрузку стенограмм, web routes и Swagger.

Тесты: API-тесты `/health`, `/analyze`, `/mcp/tool`, `/mcp/execute`, project acceptance flow.

Метрики: HTTP status, latency, успешность project flow, валидность ограничений ввода.

## Сценарий защиты

1. Открыть `/projects` и создать проект.
2. Загрузить стенограмму до `20 000` символов.
3. Запустить LLM-анализ.
4. Показать summary, решения, задачи, ответственных, риски и историю запусков.
5. Открыть `/docs` и показать REST API.
6. Открыть `/mcp/tool` и показать MCP schema.
7. Запустить `python -m pytest` и показать успешные тесты.
