import os


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.normpath(path)


def check_true(p: str) -> bool:
    if p in ("True", "true", 1, True):
        return True
    return False


def check_false(p: str) -> bool:
    if p in ("False", "false", 0, False):
        return True
    return False


def check_none(p: str) -> bool:
    if p in ("None", "none", None):
        return True
    return False


def check_for_bool(p: str) -> bool:
    if check_false(p):
        return False
    elif check_true(p):
        return True
    else:
        raise ValueError("%s is not a bool" % str(p))


def get_tuning_datasets():
    return [
        f"green_{elem}.json"
        for elem in sorted(
            [
                int((p.split(".")[0]).split("_")[1])
                for p in os.listdir(
                    os.path.join("/", "home", "interactive-mo-ml", "input")
                )
                if ".json" in p
            ]
        )
    ][:3]
