import numpy as np

from s5_classification.null_ladder import _class_names
from s5_classification.structure_only_probe import majority_class_macro_f1


def test_class_names_for_ambivalence():
    assert _class_names("stance_ambivalence") == ["low", "med", "high"]


def test_majority_class_macro_f1_matches_sklearn():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import f1_score

    y = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(y.reshape(-1, 1), y)
    expected = f1_score(y, clf.predict(y.reshape(-1, 1)), average="macro")

    assert majority_class_macro_f1(y) == expected
