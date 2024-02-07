import os
import shutil
import logging
from omegaconf import DictConfig


class RunFilter:
    def __init__(
        self,
        config: DictConfig,
        working_directory: str,
        output_directory: str
    ) -> None:
        self.config = config
        self.filters = config.get("_filters", [])
        self.working_directory = working_directory
        self.output_directory = output_directory
        self.run = os.path.basename(self.output_directory)

    def skip(self) -> bool:
        return self.should_not_run() or self.should_exclude_run()

    def should_not_run(self) -> bool:
        if os.path.exists(os.path.join(self.output_directory, "metrics.csv")):
            print(f"\nRunFilter: {self.run} already exists, skipping...\n")
            return True
        return False

    def should_exclude_run(self) -> bool:
        skip, filter = self._should_exclude_run()
        if skip:
            print(f"\nRunFilter: {self.run} filtered, due to {filter}...\n")
            for handler in logging.getLogger().handlers:
                handler.close()
            shutil.rmtree(self.output_directory)
            return True
        return False

    def _evaluate(self, conditions_str: str) -> bool:
        if " | " in conditions_str:
            split_char = " | "
            condition_check = any
        else:
            split_char = " & "
            condition_check = all
        conditions = conditions_str.split(split_char)
        conditions_eval = []
        for c in conditions:
            conditions_eval.append(eval(f"self.config.{c}"))
        if condition_check(conditions_eval):
            return True
        return False

    def _should_exclude_run(self) -> bool:
        for filter_str in self.filters:
            case_str, conditions_str = map(str.strip, filter_str.split(" ! "))
            if self._evaluate(case_str) and self._evaluate(conditions_str):
                return True, filter_str
        return False, None
