import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_features_target(df, target_col='class'):
    """
    Separates the DataFrame into features (X) and the target variable (y).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    target_col : str, optional
        The name of the target column, by default 'class'.

    Returns
    -------
    tuple
        A tuple containing X (features) and y (target).
    """
    X = df.drop(columns=target_col)
    y = df[target_col].apply(lambda x: 1 if x == '>50K' else 0)
    return X, y

def label_encode_features(X):
    """
    Applies Label Encoding to the categorical columns of X.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame of features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded categorical columns.
    """
    X_encoded = X.copy()
    cat_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
    return X_encoded

def one_hot_encode_features(X):
    """
    Applies One-Hot Encoding to the categorical columns of X.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame of features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded categorical columns.
    """
    return pd.get_dummies(X, drop_first=True)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame of features.
    y : pandas.Series
        The target series.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    tuple
        A tuple containing X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
