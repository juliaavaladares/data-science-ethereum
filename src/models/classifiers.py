from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
import joblib
import pandas as pd


def generate_models():
    knn = KNeighborsClassifier()
    decision_tree = DecisionTreeClassifier(max_depth=5)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=2)
    logistic_regression = LogisticRegression()
    linear_svm = SVC(kernel='linear', probability=True)
    gaussian_svm = SVC(kernel='rbf', probability=True)
    sigmoid_svm = SVC(kernel='sigmoid', probability=True)


    estimators = [('knn',knn), 
                  ('dct',decision_tree),
                  ('rf', random_forest), 
                  ('lr', logistic_regression), 
                  ('linear_svm', linear_svm), 
                  ('gaussian_svm', gaussian_svm),
                  ('sigmoid_svm', sigmoid_svm)]

    return estimators

def train_models(models, X_train, y_train, X_test, y_test, name, models_dir):
    """
    Treina classificadores, salva os modelos treinados e as predições de teste.

    Parameters
    ----------
    models : list of (str, estimator)
        Lista de pares (nome, modelo) a treinar.
    X_train : array-like
        Features de treino.
    y_train : array-like
        Rótulos de treino.
    X_test : array-like
        Features de teste.
    y_test : array-like
        Rótulos verdadeiros de teste.
    name : str
        Identificador usado no nome do arquivo de predições.
    models_dir : str or path-like
        Diretório onde os modelos treinados serão armazenados.

    Returns
    -------
    predictions_df : pandas.DataFrame
        DataFrame com y_true e as predições de cada modelo.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({'y_true': y_test})

    for model_name, model in models:
        model.fit(X_train, y_train)

        y_predict = model.predict(X_test)
        predictions_df[model_name] = y_predict

        joblib.dump(model, models_path / f"{model_name}_{name}")

    predictions_df.to_csv(
        f"predictions/new_data/predictions_results_{name}.csv",
        index=False,
    )

    return predictions_df
