from common import *

def task2_decision_boundary(f1=0, f2=1, n_show=15, plot=True):
    cols = [f1, f2]
    m = train_on(cols)
    w1, w2 = m.w
    print("--- Task 2: Decision Boundary Explorer ---")
    print(f"Features: {FEATURES[f1]}, {FEATURES[f2]}")
    print(f"Final weights: w1={w1:.4f}, w2={w2:.4f}   Bias: {m.b:.4f}")
    print(f"Decision boundary: {w1:.4f}*{FEATURES[f1]} + {w2:.4f}*{FEATURES[f2]} + ({m.b:.4f}) = 0")

    z = m.weighted_sum(Xte[:, cols])
    side = np.where(z > 0, 'Positive side (Forged region)',
                    np.where(z < 0, 'Negative side (Genuine region)', 'ON the boundary'))
    out = pd.DataFrame({'Actual': yte, 'Predicted': m.step(z),
                        'Weighted Sum': np.round(z, 4), 'Side of boundary': side})
    print(out.head(n_show).to_string(index=False))
    print(f"\nTest accuracy: {accuracy(m, Xte[:, cols], yte) * 100:.2f}%")

    if plot:
        plt.figure(figsize=(6, 5))
        plt.scatter(Xte[yte == 0, f1], Xte[yte == 0, f2], c='tab:blue', s=12, label='Genuine (0)')
        plt.scatter(Xte[yte == 1, f1], Xte[yte == 1, f2], c='tab:red', s=12, label='Forged (1)')
        xs = np.linspace(0, 1, 100)
        if abs(w2) > 1e-12:
            plt.plot(xs, -(w1 * xs + m.b) / w2, 'k-', label='Decision boundary')
        plt.xlim(0, 1); plt.ylim(0, 1)
        plt.xlabel(FEATURES[f1] + ' (normalized)'); plt.ylabel(FEATURES[f2] + ' (normalized)')
        plt.legend(); plt.title('Perceptron decision boundary'); plt.show()
    return m

if __name__ == '__main__':
    print('Features: ' + ', '.join(f'{i}={f}' for i, f in enumerate(FEATURES)))
    a = input('First feature index (Enter = 0): ').strip()
    b = input('Second feature index (Enter = 1): ').strip()
    f1 = int(a) if a.isdigit() and int(a) < 4 else 0
    f2 = int(b) if b.isdigit() and int(b) < 4 else 1
    if f1 == f2:
        print('Choose two different features.')
    else:
        task2_decision_boundary(f1, f2)
