import numpy as np
import pandas as pd
from perceptron import Perceptron

GATES = {
    'AND':  ([0, 0, 0, 1], 'Linearly Separable'),
    'OR':   ([0, 1, 1, 1], 'Linearly Separable'),
    'NAND': ([1, 1, 1, 0], 'Linearly Separable'),
    'XOR':  ([0, 1, 1, 0], 'Not Linearly Separable'),
}
XG = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)

def task6_xor(max_epochs=100):
    print("--- Task 6: XOR Limitation Demonstrator ---")
    summary = []
    for name, (t, expected) in GATES.items():
        t = np.array(t)
        m = Perceptron(lr=0.1, max_epochs=max_epochs).fit(XG, t, track=True)
        acc = float(np.mean(m.predict(XG) == t)) * 100
        status = 'Converged' if m.converged else f'Did NOT converge in {max_epochs} epochs'
        summary.append({'Gate': name, 'Expected': expected, 'Epochs': m.epochs_,
                        'Status': status, 'Accuracy (%)': acc})
        if name == 'XOR':
            xor_model, xor_t = m, t
    print(pd.DataFrame(summary).to_string(index=False))

    print("\nXOR repeated misclassification pattern (last 8 epochs):")
    print("Inputs order: (0,0) (0,1) (1,0) (1,1)   target:", [int(v) for v in xor_t])
    for e in range(max(0, xor_model.epochs_ - 8), xor_model.epochs_):
        p = xor_model.pred_history[e]
        wrong = [(int(XG[i][0]), int(XG[i][1])) for i in range(4) if p[i] != xor_t[i]]
        print(f"  epoch {e + 1:3d}: predictions={[int(v) for v in p]}  misclassified={wrong}  errors during epoch={xor_model.errors_per_epoch[e]}")
    print("\nExplanation: XOR outputs 1 for (0,1) and (1,0) but 0 for (0,0) and (1,1). The two classes sit on "
          "opposite diagonals of the square, so no single straight line (w1*x1 + w2*x2 + b = 0) can put both "
          "1-points on one side and both 0-points on the other. Fixing one point breaks another, so the "
          "weights keep cycling and the perceptron never reaches zero errors. A hidden layer (an MLP) is needed.")
    return summary

if __name__ == '__main__':
    task6_xor()
