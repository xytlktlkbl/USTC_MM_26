"""
传统机器学习方法对比: KNN + Random Forest vs 神经网络
"""
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), 'insects')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')


def load_data(filepath):
    data = np.loadtxt(filepath)
    X = data[:, :2].astype(np.float32)
    y = data[:, 2].astype(np.int64)
    return X, y


def load_dataset(dataset_id):
    if dataset_id == 1:
        train_file = os.path.join(DATA_DIR, "insects-training.txt")
        test_file = os.path.join(DATA_DIR, "insects-testing.txt")
    else:
        train_file = os.path.join(DATA_DIR, "insects-2-training.txt")
        test_file = os.path.join(DATA_DIR, "insects-2-testing.txt")
    X_train, y_train = load_data(train_file)
    X_test, y_test = load_data(test_file)
    return X_train, y_train, X_test, y_test


def plot_decision_boundary(model, X, y, scaler, title, save_path):
    """通用决策边界绘图"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_scaled = scaler.transform(grid)
    Z = model.predict(grid_scaled).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ['#FF0000', '#00AA00', '#0000FF']
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    for c in range(3):
        mask = y == c
        plt.scatter(X[mask, 0], X[mask, 1], c=cmap_bold[c], label=f'Class {c}',
                    edgecolors='k', s=30)
    plt.xlabel('Body Length')
    plt.ylabel('Wing Length')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []

    # KNN: 不同 k 值
    for ds_id, ds_name in [(1, 'Dataset 1'), (2, 'Dataset 2')]:
        X_train, y_train, X_test, y_test = load_dataset(ds_id)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for k in [1, 3, 5, 7, 9, 11]:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train_s, y_train)

            train_acc = accuracy_score(y_train, knn.predict(X_train_s))
            test_acc_all = accuracy_score(y_test, knn.predict(X_test_s))
            test_acc_first60 = accuracy_score(y_test[:60], knn.predict(X_test_s[:60]))
            test_acc_last150 = accuracy_score(y_test[60:], knn.predict(X_test_s[60:]))

            results.append({
                'method': 'KNN',
                'dataset': ds_id,
                'param': f'k={k}',
                'train_acc': train_acc,
                'test_acc_all': test_acc_all,
                'test_acc_first60': test_acc_first60,
                'test_acc_last150': test_acc_last150,
            })

            # 画最优 k 的决策边界
            if k == 5:
                plot_decision_boundary(knn, X_test, y_test, scaler,
                    f'KNN (k={k}) Decision Boundary — {ds_name} (Test)',
                    os.path.join(OUTPUT_DIR, f'dataset{ds_id}_knn_k{k}_decision_test.png'))

    # Random Forest: 不同 n_estimators 和 max_depth
    for ds_id, ds_name in [(1, 'Dataset 1'), (2, 'Dataset 2')]:
        X_train, y_train, X_test, y_test = load_dataset(ds_id)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        configs = [
            (50, None), (100, None), (200, None),
            (100, 5), (100, 10), (100, 15),
        ]
        for n_est, max_d in configs:
            param_str = f'n={n_est}' + (f',depth={max_d}' if max_d else '')
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                        random_state=42)
            rf.fit(X_train_s, y_train)

            train_acc = accuracy_score(y_train, rf.predict(X_train_s))
            test_acc_all = accuracy_score(y_test, rf.predict(X_test_s))
            test_acc_first60 = accuracy_score(y_test[:60], rf.predict(X_test_s[:60]))
            test_acc_last150 = accuracy_score(y_test[60:], rf.predict(X_test_s[60:]))

            results.append({
                'method': 'RF',
                'dataset': ds_id,
                'param': param_str,
                'train_acc': train_acc,
                'test_acc_all': test_acc_all,
                'test_acc_first60': test_acc_first60,
                'test_acc_last150': test_acc_last150,
            })

            # 画最优配置的决策边界
            if n_est == 100 and max_d is None:
                plot_decision_boundary(rf, X_test, y_test, scaler,
                    f'Random Forest ({param_str}) Decision Boundary — {ds_name} (Test)',
                    os.path.join(OUTPUT_DIR, f'dataset{ds_id}_rf_{param_str.replace(",","_").replace("=","")}_decision_test.png'))

    # 打印汇总
    print(f"{'Method':<8} {'Dataset':<8} {'Param':<18} {'Train':>8} {'TestAll':>8} {'前60':>8} {'后150':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['method']:<8} {r['dataset']:<8} {r['param']:<18} "
              f"{r['train_acc']:>8.4f} {r['test_acc_all']:>8.4f} "
              f"{r['test_acc_first60']:>8.4f} {r['test_acc_last150']:>8.4f}")

    # 保存 JSON
    with open(os.path.join(OUTPUT_DIR, 'traditional_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # 保存文本汇总
    with open(os.path.join(OUTPUT_DIR, 'traditional_summary.txt'), 'w') as f:
        for r in results:
            f.write(f"{r['method']},{r['dataset']},{r['param']},"
                    f"{r['train_acc']:.4f},{r['test_acc_all']:.4f},"
                    f"{r['test_acc_first60']:.4f},{r['test_acc_last150']:.4f}\n")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
