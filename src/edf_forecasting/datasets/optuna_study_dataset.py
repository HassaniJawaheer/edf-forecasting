from kedro.io import AbstractVersionedDataset
from typing import Any, Dict, Optional
import optuna
import shutil
from pathlib import Path

class OptunaStudyDataset(AbstractVersionedDataset):
    def __init__(self, filepath: str, study_name: str, version: Optional[str] = None):
        super().__init__(filepath, version)
        self._filepath = Path(filepath)
        self._filename = self._filepath.name
        self._base_dir = self._filepath.parent
        self._study_name = study_name

    def _load(self) -> optuna.Study:
        load_path = self._get_load_path().with_name(self._filename)
        storage = f"sqlite:///{load_path}"
        return optuna.load_study(study_name=self._study_name, storage=storage)

    def _save(self, study: optuna.Study) -> None:
        save_path = self._get_save_path().with_name(self._filename)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._filepath, save_path)

    def _describe(self) -> Dict[str, Any]:
        return {
            "filepath": str(self._filepath),
            "study_name": self._study_name,
            "version": str(self._version) if self._version else None,
        }

