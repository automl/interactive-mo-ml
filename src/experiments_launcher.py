import json
import os
import copy
import subprocess
from tqdm import tqdm

from yahpo_gym import local_config
from yahpo_gym import benchmark_set
import yahpo_gym.benchmarks.lcbench

from utils.common import get_tuning_datasets, make_dir


def run_cmd(cmd, stdout_path, stderr_path):
    open(stdout_path, "w")
    open(stderr_path, "w")
    with open(stdout_path, "a") as log_out:
        with open(stderr_path, "a") as log_err:
            subprocess.call(
                cmd,
                shell=True,
                stdout=log_out,
                stderr=log_err,
            )


if __name__ == "__main__":
    input_path = make_dir(os.path.join("/", "home", "interactive-mo-ml", "input"))
    common_log_path = make_dir(os.path.join("/", "home", "interactive-mo-ml", "logs"))
    log_paths = {
        "preliminar_sampling": make_dir(
            os.path.join(common_log_path, "preliminar_sampling")
        ),
        "automatic_ordering": make_dir(
            os.path.join(common_log_path, "automatic_ordering")
        ),
        "preference_learning": make_dir(
            os.path.join(common_log_path, "preference_learning_eval")
        ),
        "optimization": make_dir(os.path.join(common_log_path, "optimization")),
        "comparison": make_dir(os.path.join(common_log_path, "comparison")),
        "summarizer": make_dir(os.path.join(common_log_path, "summarizer")),
        "plotter": make_dir(os.path.join(common_log_path, "plotter")),
    }

    print("\n\n")
    print("--- SCENARIO GENERATION ---")
    subprocess.call("python src/scenario_generator.py", shell=True)
    confs = [p for p in os.listdir(input_path) if ".json" in p]

    print("\n\n")
    print("--- PRELIMINARY SAMPLING ---")
    with tqdm(total=len(confs)) as pbar:
        for conf in confs:
            log_file_name = conf.split(".")[0]
            run_cmd(
                f"python src/preliminar_sampling.py --conf_file {conf}",
                os.path.join(
                    log_paths["preliminar_sampling"], f"{log_file_name}_stdout.txt"
                ),
                os.path.join(
                    log_paths["preliminar_sampling"], f"{log_file_name}_stderr.txt"
                ),
            )

            run_cmd(
                f"python src/automatic_ordering.py --conf_file {conf}",
                os.path.join(
                    log_paths["automatic_ordering"], f"{log_file_name}_stdout.txt"
                ),
                os.path.join(
                    log_paths["automatic_ordering"], f"{log_file_name}_stderr.txt"
                ),
            )

            pbar.update()

    print("\n\n")
    print("--- INTERACTIVE PREFERENCE LEARNING ---")
    print("\tIt might take some minutes...")
    run_cmd(
        "python src/preference_learning_eval.py",
        os.path.join(log_paths["preliminar_sampling"], f"stdout.txt"),
        os.path.join(log_paths["preliminar_sampling"], f"stderr.txt"),
    )
    print("DONE.")

    evaluation_confs = [elem for elem in confs if elem not in get_tuning_datasets()]

    print("\n\n")
    print("--- UTILITY-DRIVEN HPO ---")
    with tqdm(total=len(evaluation_confs)) as pbar:
        for conf in evaluation_confs:
            log_file_name = conf.split(".")[0]
            run_cmd(
                f"python src/optimization.py --conf_file {conf}",
                os.path.join(log_paths["optimization"], f"{log_file_name}_stdout.txt"),
                os.path.join(log_paths["optimization"], f"{log_file_name}_stderr.txt"),
            )
            run_cmd(
                f"python src/comparison.py --conf_file {conf}",
                os.path.join(log_paths["comparison"], f"{log_file_name}_stdout.txt"),
                os.path.join(log_paths["comparison"], f"{log_file_name}_stderr.txt"),
            )
            pbar.update()

    print("\n\n")
    print("--- RESULT COLLECTION ---")
    run_cmd(
        f"python src/summarizer.py --conf_file {conf}",
        os.path.join(log_paths["summarizer"], f"{log_file_name}_stdout.txt"),
        os.path.join(log_paths["summarizer"], f"{log_file_name}_stderr.txt"),
    )
    run_cmd(
        f"python src/plotter.py --conf_file {conf}",
        os.path.join(log_paths["plotter"], f"{log_file_name}_stdout.txt"),
        os.path.join(log_paths["plotter"], f"{log_file_name}_stderr.txt"),
    )
