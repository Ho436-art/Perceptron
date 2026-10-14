from common import *

def task4_sample_order():
    n = len(ytr)
    rng = np.random.default_rng(SEED)
    orders = {
        'Original': np.arange(n),
        'Reverse': np.arange(n)[::-1],
        'Random': rng.permutation(n),
        'Class-wise grouped': np.argsort(ytr, kind='stable'),   # all class 0 first, then all class 1
    }
    rows, models = [], {}
    for name, idx in orders.items():
        m = train_on(ALL, order=idx)
        models[name] = m
        rows.append({'Order': name, 'Epochs': m.epochs_, 'Converged': m.converged,
                     'Weights': np.round(m.w, 4), 'Bias': round(m.b, 4),
                     'Accuracy (%)': round(accuracy(m, Xte[:, ALL], yte) * 100, 2)})
    res = pd.DataFrame(rows)
    print("--- Task 4: Training Sample Order Challenge ---")
    print(res.to_string(index=False))

    base = models['Original']
    different = any(not (np.allclose(base.w, m.w) and np.isclose(base.b, m.b)) for m in models.values())
    print("\nDo different sample orders produce different decision boundaries?",
          "YES - weights/bias differ between orders." if different else "NO - identical boundaries.")
    print("The perceptron updates after every sample, so the order of samples changes the path of updates "
          "and usually the final boundary, even though many boundaries can separate the data equally well.")
    return res

if __name__ == '__main__':
    task4_sample_order()
