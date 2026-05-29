from app.api.evaluation.pipeline.deterministic_scorer import score_expected_vs_actual


def test_scorer_normalizes_task_structured_fields():
    expected = {
        "tasks": [
            {
                "description": "Проверить отчёт по аудиту вентиляции",
                "assignee": "Алексей",
                "status": "в работе",
                "priority": "критический",
                "due_date": "29.05.2026",
            }
        ]
    }
    actual = {
        "tasks": [
            {
                "description": "Алексей проверит отчет по аудиту вентиляции",
                "assignee": "алексей",
                "status": "in_progress",
                "priority": "high",
                "due_date": "2026-05-29",
            }
        ]
    }

    score = score_expected_vs_actual(expected, actual)
    task_score = score["categories"]["tasks"]

    assert task_score["matched"] == 1
    assert task_score["field_accuracy"] == {
        "assignee": 1.0,
        "status": 1.0,
        "priority": 1.0,
        "due_date": 1.0,
    }


def test_scorer_repairs_common_mojibake_for_matching():
    expected = {
        "risks": [
            {"description": "Риск остановки производства из-за износа подшипников"}
        ]
    }
    actual = {
        "risks": [
            "Р РёСЃРє РѕСЃС‚Р°РЅРѕРІРєРё РїСЂРѕРёР·РІРѕРґСЃС‚РІР° РёР·-Р·Р° РёР·РЅРѕСЃР° РїРѕРґС€РёРїРЅРёРєРѕРІ"
        ]
    }

    score = score_expected_vs_actual(expected, actual)

    assert score["categories"]["risks"]["matched"] == 1


def test_scorer_uses_token_overlap_for_reordered_text():
    expected = {
        "decisions": [
            {"description": "Перенести новый дизайн профиля после стабилизации текущей версии"}
        ]
    }
    actual = {
        "decisions": [
            "Сначала стабилизировать текущую версию, дизайн профиля перенести"
        ]
    }

    score = score_expected_vs_actual(expected, actual)

    assert score["categories"]["decisions"]["matched"] == 1


def test_scorer_scores_task_update_due_dates():
    expected = {
        "task_updates": [
            {
                "description": "Перенести аудит вентиляции",
                "assignee": "Алексей",
                "status": "in_progress",
                "due_date": "19.06.2026",
            }
        ]
    }
    actual = {
        "task_updates": [
            {
                "description": "Аудит вентиляции перенесен",
                "assignee": "Алексей",
                "status": "в работе",
                "due_date": "2026-06-19",
            }
        ]
    }

    score = score_expected_vs_actual(expected, actual)
    update_score = score["categories"]["task_updates"]

    assert update_score["matched"] == 1
    assert update_score["field_accuracy"]["due_date"] == 1.0
