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

