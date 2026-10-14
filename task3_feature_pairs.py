from common import *

def task3_feature_pairs():
    rows = []
    for a, b in itertools.combinations(range(4), 2):
        m = train_on([a, b])
        pred = m.predict(Xte[:, [a, b]])
        rows.append({
            'Feature Pair': f"{FEATURES[a]} + {FEATURES[b]}",
            'Epochs': m.epochs_,
            'Converged': m.converged,
            'Accuracy (%)': round(float(np.mean(pred == yte)) * 100, 2),
            'Misclassified (test)': int(np.sum(pred != yte)),
        })
    res = pd.DataFrame(rows).sort_values(
        ['Accuracy (%)', 'Misclassified (test)', 'Epochs'],
        ascending=[False, True, True]).reset_index(drop=True)
    res.index += 1
    res.index.name = 'Rank'
    print("--- Task 3: Feature Pair Separability (ranked, best first) ---")
    print(res.to_string())
    print("\nMost linearly separable feature pair:", res.iloc[0]['Feature Pair'])
    return res

if __name__ == '__main__':
    task3_feature_pairs()
