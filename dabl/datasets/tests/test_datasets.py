import pandas as pd
from dabl.datasets import load_ames, load_titanic, load_adult, data_path


def test_loading():
    # smoke test
    ames = load_ames()
    assert ames.shape == (2930, 82)
    titanic = load_titanic()
    assert titanic.shape == (1309, 14)
    adult = load_adult()
    assert adult.shape == (32561, 15)

    pd.read_csv(data_path("titanic.csv"))
