from common import *

def task1_banknote_classification(n_show=15):
    m = train_on(ALL)
    z = m.weighted_sum(Xte[:, ALL])
    pred = m.step(z)
    out = pd.DataFrame({
        'Actual Class': yte,
        'Predicted Class': pred,
        'Weighted Sum': np.round(z, 4),
        'Label': np.where(yte == 1, 'Forged', 'Genuine')
    })
    print("--- Task 1: Banknote Classification ---")
    print(f"Epochs run: {m.epochs_}   Converged: {m.converged}")
    print("Weights:", np.round(m.w, 4), " Bias:", round(m.b, 4))
    print(out.head(n_show).to_string(index=False))
    print(f"\nTrain accuracy: {accuracy(m, Xtr[:, ALL], ytr) * 100:.2f}%")
    print(f"Test accuracy : {accuracy(m, Xte[:, ALL], yte) * 100:.2f}%")
    return m

if __name__ == '__main__':
    task1_banknote_classification()
