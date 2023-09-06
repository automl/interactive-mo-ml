def get_element_from_results(results, preference_budget, column, row, mode, aggregate):
    return round(
        results.loc[
            (results["second_indicator"] == column)
            & (results["preference_budget"] == preference_budget)
            & (results["main_indicator"] == (column if mode == "preferences" else row))
            & (results["mode"] == mode),
            f"indicators {aggregate}",
        ].values[0],
        2,
    )


def get_cell_value(results, preference_budget, column, row):
    def build_string(mode):
        return f"""{temp_results[mode]["mean"]}  (+-{temp_results[mode]["std"]})"""

    temp_results = {
        mode: {
            aggregate: get_element_from_results(
                results, preference_budget, column, row, mode, aggregate
            )
            for aggregate in ["mean", "std"]
        }
        for mode in ["indicators", "preferences"]
    }

    return f"""{build_string("indicators")} / {build_string("preferences")}"""
