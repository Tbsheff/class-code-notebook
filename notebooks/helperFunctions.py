def univariate(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df_output = pd.DataFrame(
        columns=[
            "type",
            "missing",
            "unique",
            "min",
            "q1",
            "median",
            "q3",
            "max",
            "mean",
            "mode",
            "std",
            "skew",
            "kurt",
        ]
    )

    for col in df:
        # Stuff that applies to all data types
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            # Stuff that only applies to numeric dtypes
            min = df[col].min()
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            max = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()
            kurtosis = df[col].kurt()
            df_output.loc[col] = [
                "numeric",
                missing,
                unique,
                min,
                q1,
                median,
                q3,
                max,
                mean,
                mode,
                std,
                skew,
                kurtosis,
            ]

            ax = sns.histplot(data=df, x=col)
            plt.show()
        else:
            df_output.loc[col] = [
                "categorical",
                missing,
                unique,
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                mode,
                "-",
                "-",
                "-",
            ]
            ax = sns.countplot(data=df, x=col)
            plt.show()
    return df_output


def basic_wrangle(df, unique_thresh=0.95, na_thresh=0.5, messages=True):
    import pandas as pd

    for col in df:
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        rows = df.shape[0]
        if messages:
            print(
                f"{col} has {missing} missing values and {unique} unique values. ({rows} rows total)"
            )
        # Drop columns with more than na_thresh missing values
        if missing / rows > na_thresh:
            df = df.drop(columns=col, inplace=True)
            if messages:
                print(f"Dropped {col} due to missing values")
        # Drop columns with a high ratio of unique values
        elif unique / rows > unique_thresh:
            df = df.drop(columns=col, inplace=True)
            if messages:
                print(f"Dropped {col} due to high ratio of unique values")
        # Drop columns with only 1 unique value
        elif unique == 1:
            df = df.drop(columns=col, inplace=True)
            if messages:
                print(
                    f"Dropped {col} due to only 1 unique value, {df[col].unique()[0]}"
                )

    return df


def regression(df, label):
    import pandas as pd
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        LassoLars,
        BayesianRidge,
        TweedieRegressor,
    )

    y = df[label]
    X = df.drop(columns=label)

    # OLS
    model_ols = LinearRegression().fit(X, y)
    score_ols = model_ols.score(X, y)
    print(f"Score OLS: \t{score_ols}")

    # Ridge
    model_ridge = Ridge(alpha=0.5).fit(X, y)
    score_ridge = model_ridge.score(X, y)
    print(f"Score Ridge: \t{score_ridge}")

    # Lasso
    model_lasso = Lasso(alpha=2).fit(X, y)
    score_lasso = model_lasso.score(X, y)
    print(f"Score Lasso: \t{score_lasso}")

    # Lasso Lars
    model_lasso_lars = LassoLars(alpha=2).fit(X, y)
    score_lasso_lars = model_lasso_lars.score(X, y)
    print(f"Score Lasso Lars: \t{score_lasso_lars}")

    # Baysian Ridge
    model_Bayesian_Ridge = BayesianRidge().fit(X, y)
    score_bayesian_ridge = model_Bayesian_Ridge.score(X, y)
    print(f"Score Bayesian Ridge: \t{score_bayesian_ridge}")

    # Tweedie Regressor power 1
    model_tr1 = TweedieRegressor(power=1, link="log", max_iter=1000).fit(X, y)
    score_tr1 = model_tr1.score(X, y)
    print(f"Score Tweedie Regressor 1: \t{score_tr1}")

    # Tweedie Regressor power 2
    model_tr2 = TweedieRegressor(power=2, link="log", max_iter=1000).fit(X, y)
    score_tr2 = model_tr2.score(X, y)
    print(f"Score Tweedie Regressor 2: \t{score_tr2}")

    # Tweedie Regressor power 3
    model_tr3 = TweedieRegressor(power=3, link="log", max_iter=1000).fit(X, y)
    score_tr3 = model_tr3.score(X, y)
    print(f"Score Tweedie Regressor 3: \t{score_tr3}")

    return (
        model_ols,
        model_ridge,
        model_lasso,
        model_lasso_lars,
        model_Bayesian_Ridge,
        model_tr1,
        model_tr2,
        model_tr3,
    )
