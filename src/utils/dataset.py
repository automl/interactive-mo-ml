import openml


def load_dataset_from_openml(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    # cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    # Xt = pd.DataFrame(X)
    # Xt[cat_features] = Xt[cat_features].fillna(-1)
    # Xt[cat_features] = Xt[cat_features].astype("str")
    # Xt[cat_features] = Xt[cat_features].replace("-1", np.nan)
    # Xt = Xt.to_numpy()
    # return Xt, y, categorical_indicator
    return X, y, categorical_indicator, feature_names
