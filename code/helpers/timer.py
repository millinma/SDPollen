import datetime
import time
import yaml
import os


class Timer:
    def __init__(self, output_directory: str, timer_type: str) -> None:
        self.time_log = []
        self.start_time = None
        self.output_directory = output_directory
        self.timer_type = timer_type

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> None:
        if self.start_time is None:
            raise Exception("Timer not yet started!")
        run_time = time.time() - self.start_time
        self.start_time = None
        self.time_log.append(run_time)

    def get_time_log(self) -> list:
        return self.time_log

    def get_mean_seconds(self) -> float:
        return sum(self.time_log)/len(self.time_log)

    def get_total_seconds(self) -> float:
        return sum(self.time_log)

    @classmethod
    def pretty_time(cls, seconds: float) -> str:
        pretty = datetime.timedelta(seconds=int(seconds))
        return str(pretty)

    def save(self, path="") -> dict:
        out_path = os.path.join(self.output_directory, path, "timer.yaml")
        time_dict = {
            self.timer_type: {
                "mean": Timer.pretty_time(self.get_mean_seconds()),
                "mean_seconds": self.get_mean_seconds(),
                "total": Timer.pretty_time(self.get_total_seconds()),
                "total_seconds": self.get_total_seconds(),
            }
        }
        with open(out_path, "a") as f:
            yaml.dump(time_dict, f)
