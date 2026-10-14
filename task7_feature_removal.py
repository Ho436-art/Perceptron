from common import *

def task7_feature_removal():
    base = train_on(ALL)
    base_acc = accuracy(base, Xte[:, ALL], yte) * 100
    rows = []
    for rem in ALL:
        cols = [c for c in ALL if c != rem]
        m = train_on(cols)
        acc = accuracy(m, Xte[:, cols], yte) * 100
        rows.append({'Removed Feature': FEATURES[rem], 'Accuracy (%)': round(acc, 2),
                     'Accuracy Drop (%)': round(base_acc - acc, 2), 'Epochs': m.epochs_})
    res = pd.DataFrame(rows).sort_values('Accuracy Drop (%)', ascending=False).reset_index(drop=True)
    print("--- Task 7: Feature Removal Challenge ---")
    print(f"Baseline accuracy (all features): {base_acc:.2f}%")
    print(res.to_string(index=False))
    print("\nMost important feature (largest drop when removed):", res.iloc[0]['Removed Feature'])
    return res

if __name__ == '__main__':
    task7_feature_removal()
