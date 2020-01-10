from svm import SVM

data1 = [
    [-0.4326, 1.1909],
    [3.0, 4.0],
    [0.1253, -0.0376],
    [0.2877, 0.3273],
    [-1.1465, 0.1746],
    [1.8133, 2.1139],
    [2.7258, 3.0668],
    [1.4117, 2.0593],
    [4.1832, 1.9044],
    [1.8636, 1.1677],
]
labels1 = [
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
]


def test_results1():
    svm = SVM()
    iterations = svm.train(data1, labels1, {"kernel": "rbf"})
    print(iterations)
    res = svm.predict(data1)
    assert (res == labels1)


def test_restults2():
    options = {
        "kernel": "rbf",
        "rbf_sigma": 0.5
    }
    svm = SVM()
    svm.train([[0, 0], [0, 1], [1, 0], [1, 1]], [-1, 1, 1, -1], options)
    assert (svm.predict([[0, 0]]) == -1)
    assert (svm.predict([[0, 1]]) == 1)
    assert (svm.predict([[1, 0]]) == 1)
    assert (svm.predict([[1, 1]]) == -1)
    json = svm.to_json()

    svm2 = SVM()
    assert (svm2.predict([[0, 1]]) == -1)

    svm2.from_json(json)
    assert (svm2.predict([[0, 0]]) == -1)
    assert (svm2.predict([[0, 1]]) == 1)
    assert (svm2.predict([[1, 0]]) == 1)
    assert (svm2.predict([[1, 1]]) == -1)
