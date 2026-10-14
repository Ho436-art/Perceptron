from common import *

def task5_borderline(top_k=5):
    m = train_on(ALL)
    z = m.weighted_sum(Xte[:, ALL])
    abs_z = np.abs(z)                                # absolute weighted sum, before step activation
    dist = distance_to_boundary(m, Xte[:, ALL])      # |w.x + b| / ||w||
    order = np.argsort(abs_z)[:top_k]
    out = pd.DataFrame(Xte[order], columns=FEATURES).round(4)
    out.insert(0, 'Test Index', order)
    out['|Weighted Sum|'] = abs_z[order].round(5)
    out['Predicted Class'] = m.step(z[order])
    out['Actual Class'] = yte[order]
    out['Distance to Boundary'] = dist[order].round(5)
    print("--- Task 5: Borderline Sample Detector ---")
    print(f"Top {top_k} borderline test samples (closest to the decision threshold):")
    print(out.to_string(index=False))
    print(f"\nBorderline samples overall (distance < {BORDER_EPS}): {int(np.sum(dist < BORDER_EPS))}")
    return out

if __name__ == '__main__':
    task5_borderline()
