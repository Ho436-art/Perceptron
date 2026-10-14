from common import *
from perceptron import Perceptron
from task6_xor import GATES, XG

def task8_report():
    n = len(ytr)
    rng = np.random.default_rng(SEED)
    rows = []

    def add(exp, cols_names, order_name, m, Xe, ye):
        dist = distance_to_boundary(m, Xe)
        rows.append({'Experiment': exp, 'Features': cols_names, 'Sample Order': order_name,
                     'Epochs': m.epochs_, 'Weights': np.round(m.w, 3), 'Bias': round(m.b, 3),
                     'Accuracy (%)': round(float(np.mean(m.predict(Xe) == ye)) * 100, 2),
                     'Borderline Samples': int(np.sum(dist < BORDER_EPS)),
                     'Converged': m.converged})

    # Exp 1: all features
    m = train_on(ALL); add('1. Classification', 'all 4', 'Original', m, Xte[:, ALL], yte)
    # Exp 2/3: feature pairs
    for a, b in itertools.combinations(range(4), 2):
        m = train_on([a, b]); add('2/3. Feature pair', f"{FEATURES[a]}, {FEATURES[b]}", 'Original', m, Xte[:, [a, b]], yte)
    # Exp 4: sample orders
    orders = {'Original': np.arange(n), 'Reverse': np.arange(n)[::-1],
              'Random': rng.permutation(n), 'Class-wise grouped': np.argsort(ytr, kind='stable')}
    for name, idx in orders.items():
        m = train_on(ALL, order=idx); add('4. Sample order', 'all 4', name, m, Xte[:, ALL], yte)
    # Exp 6: logic gates
    for name, (t, _) in GATES.items():
        t = np.array(t); m = Perceptron(0.1, 100).fit(XG, t)
        add('6. Logic gate ' + name, 'x1, x2', 'Original', m, XG, t)
    # Exp 7: feature removal
    for rem in ALL:
        cols = [c for c in ALL if c != rem]
        m = train_on(cols); add('7. Remove ' + FEATURES[rem], ', '.join(FEATURES[c] for c in cols), 'Original', m, Xte[:, cols], yte)

    rep = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
    print("--- Task 8: Perceptron Classification Report ---")
    print(rep.to_string(index=False))
    return rep

if __name__ == '__main__':
    task8_report()
