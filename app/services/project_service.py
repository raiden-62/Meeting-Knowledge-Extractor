from sqlalchemy.orm import Session

from app.db import models


def delete_project(db: Session, project_id: int) -> bool:
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        return False

    db.delete(project)
    db.commit()
    return True
