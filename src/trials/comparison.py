import pandas as pd

prova = pd.DataFrame(
    {
        "main_indicator": "r2",
        "mode": "preferences",
        "preference_budget": "140",
        "indicators": {
            "hv": 17.412119884600777,
            "sp": 0.12012637669524844,
            "ms": 0.2103660475491107,
            "r2": 0.7480296325683594,
        },
        "preferences": {
            "hv": 11.257261748358301,
            "sp": 1.5399459182025588,
            "ms": -2.666964632297257,
            "r2": -0.11367675223091434,
        },
    }
)
print(prova)
prova = pd.concat(
    [
        prova,
        pd.DataFrame(
            {
                "main_indicator": "hv",
                "mode": "indicators",
                "preference_budget": "28",
                "indicators": {
                    "hv": 17.412119884600777,
                    "sp": 0.12012637669524844,
                    "ms": 0.2103660475491107,
                    "r2": 0.7480296325683594,
                },
                "preferences": {
                    "hv": 11.257261748358301,
                    "sp": 1.5399459182025588,
                    "ms": -2.666964632297257,
                    "r2": -0.11367675223091434,
                },
            }
        ),
    ]
)
print(prova)
