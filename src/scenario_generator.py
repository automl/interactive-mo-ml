import json
import os
import copy
from tqdm import tqdm

from yahpo_gym import local_config
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from utils.common import make_dir

template = {
    "model": "lcbench",
    "use_case": "green_automl",
    "performance_objective": {"metric": "Accuracy", "mode": "max"},
    "use_case_objective": {"metric": "Power consumption (W)", "mode": "min"},
    "grid_samples": 10,
    "random_samples": 40,
    "preference_samples": 500,
    "optimization_samples": 30,
    "seed": 0,
}

if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "interactive-mo-ml", "input"))
    local_config.init_config()
    local_config.set_data_path("/home/yahpo_data-1.0")
    tasks = benchmark_set.BenchmarkSet("lcbench").instances
    with tqdm(total=len(tasks)) as pbar:
        for task in tasks:
            task_template = copy.deepcopy(template)
            task_template["task"] = task
            task_template["output_folder"] = f"green_{task}"
            with open(os.path.join(input_path, f"green_{task}.json"), "w") as f:
                json.dump(task_template, f)
            pbar.update()
