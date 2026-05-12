"""
超参数对比实验 — 测试不同网络结构、激活函数等对分类效果的影响
"""
import subprocess
import os
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
SCRIPT = os.path.join(os.path.dirname(__file__), 'train.py')
PYTHON_EXE = r"H:\Envs\bio\Scripts\python.exe"


def run_experiment(dataset, hidden_layers, activation, dropout, epochs, lr, wd, seed=42):
    """运行单次实验，返回结果字典"""
    cmd = [
        PYTHON_EXE, SCRIPT,
        '--dataset', str(dataset),
        '--hidden-layers'] + [str(h) for h in hidden_layers] + [
        '--activation', activation,
        '--dropout', str(dropout),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--weight-decay', str(wd),
        '--seed', str(seed),
        '--output-dir', OUTPUT_DIR,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "output": ""}

    # 从输出中解析结果
    lines = output.split('\n')
    res = {
        "train_acc": None,
        "test_acc_all": None,
        "test_acc_first60": None,
        "test_acc_last150": None,
    }
    for line in lines:
        line = line.strip()
        if line.startswith("Train Accuracy:"):
            res["train_acc"] = float(line.split(":")[1].strip())
        elif line.startswith("Test Accuracy (all):"):
            res["test_acc_all"] = float(line.split(":")[1].strip())
        elif line.startswith("Test Accuracy (前60):"):
            res["test_acc_first60"] = float(line.split(":")[1].strip().replace('前60):', '').strip())
        elif "前60" in line and "Test Accuracy" in line:
            res["test_acc_first60"] = float(line.split(":")[-1].strip())
        elif "后150" in line and "Test Accuracy" in line:
            res["test_acc_last150"] = float(line.split(":")[-1].strip())

    res["output"] = output
    return res


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    experiments = []

    # ============================================================
    # 实验1: 不同网络结构 (数据集1)
    # ============================================================
    print("=" * 70)
    print("实验1: 不同网络结构 (Dataset 1, activation=relu)")
    print("=" * 70)

    archs = [
        ([8], "单隐藏层-8"),
        ([16], "单隐藏层-16"),
        ([32], "单隐藏层-32"),
        ([64], "单隐藏层-64"),
        ([16, 8], "双隐藏层-16-8"),
        ([32, 16], "双隐藏层-32-16"),
        ([64, 32], "双隐藏层-64-32"),
        ([128, 64], "双隐藏层-128-64"),
        ([32, 16, 8], "三隐藏层-32-16-8"),
    ]

    for hidden, name in archs:
        print(f"\n  Testing: {name}...")
        res = run_experiment(1, hidden, 'relu', 0.0, 500, 0.01, 1e-4)
        res['experiment'] = '架构对比'
        res['name'] = name
        res['dataset'] = 1
        res['hidden'] = str(hidden)
        res['activation'] = 'relu'
        experiments.append(res)
        if res.get('test_acc_all') is not None:
            print(f"    Train: {res['train_acc']:.4f}, Test(all): {res['test_acc_all']:.4f}, "
                  f"前60: {res.get('test_acc_first60', 'N/A')}, 后150: {res.get('test_acc_last150', 'N/A')}")
        else:
            print(f"    FAILED: {res.get('error', 'unknown')}")

    # ============================================================
    # 实验2: 不同激活函数 (数据集1)
    # ============================================================
    print("\n" + "=" * 70)
    print("实验2: 不同激活函数 (Dataset 1, hidden=[32,16])")
    print("=" * 70)

    activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
    for act in activations:
        print(f"\n  Testing: activation={act}...")
        res = run_experiment(1, [32, 16], act, 0.0, 500, 0.01, 1e-4)
        res['experiment'] = '激活函数对比'
        res['name'] = f'activation={act}'
        res['dataset'] = 1
        res['hidden'] = '[32, 16]'
        res['activation'] = act
        experiments.append(res)
        if res.get('test_acc_all') is not None:
            print(f"    Train: {res['train_acc']:.4f}, Test(all): {res['test_acc_all']:.4f}, "
                  f"前60: {res.get('test_acc_first60', 'N/A')}, 后150: {res.get('test_acc_last150', 'N/A')}")
        else:
            print(f"    FAILED: {res.get('error', 'unknown')}")

    # ============================================================
    # 实验3: Dropout影响 (数据集1)
    # ============================================================
    print("\n" + "=" * 70)
    print("实验3: Dropout 影响 (Dataset 1, hidden=[64,32], relu)")
    print("=" * 70)

    for do in [0.0, 0.1, 0.3, 0.5]:
        print(f"\n  Testing: dropout={do}...")
        res = run_experiment(1, [64, 32], 'relu', do, 500, 0.01, 1e-4)
        res['experiment'] = 'Dropout对比'
        res['name'] = f'dropout={do}'
        res['dataset'] = 1
        res['hidden'] = '[64, 32]'
        res['activation'] = 'relu'
        experiments.append(res)
        if res.get('test_acc_all') is not None:
            print(f"    Train: {res['train_acc']:.4f}, Test(all): {res['test_acc_all']:.4f}, "
                  f"前60: {res.get('test_acc_first60', 'N/A')}, 后150: {res.get('test_acc_last150', 'N/A')}")
        else:
            print(f"    FAILED: {res.get('error', 'unknown')}")

    # ============================================================
    # 实验4: 数据集1 vs 数据集2 (最佳配置)
    # ============================================================
    print("\n" + "=" * 70)
    print("实验4: 数据集1 vs 数据集2 (hidden=[32,16], relu)")
    print("=" * 70)

    for ds in [1, 2]:
        print(f"\n  Testing: Dataset {ds}...")
        res = run_experiment(ds, [32, 16], 'relu', 0.0, 500, 0.01, 1e-4)
        res['experiment'] = '数据集对比'
        res['name'] = f'Dataset {ds}'
        res['dataset'] = ds
        res['hidden'] = '[32, 16]'
        res['activation'] = 'relu'
        experiments.append(res)
        if res.get('test_acc_all') is not None:
            print(f"    Train: {res['train_acc']:.4f}, Test(all): {res['test_acc_all']:.4f}, "
                  f"前60: {res.get('test_acc_first60', 'N/A')}, 后150: {res.get('test_acc_last150', 'N/A')}")
        else:
            print(f"    FAILED: {res.get('error', 'unknown')}")

    # ============================================================
    # 实验5: 含噪数据集上不同架构
    # ============================================================
    print("\n" + "=" * 70)
    print("实验5: 含噪数据集上不同架构 (Dataset 2)")
    print("=" * 70)

    for hidden, name in [([16], "单层-16"), ([32, 16], "双层-32-16"), ([64, 32], "双层-64-32")]:
        print(f"\n  Testing: {name} on Dataset 2...")
        res = run_experiment(2, hidden, 'relu', 0.0, 500, 0.01, 1e-4)
        res['experiment'] = '含噪数据架构对比'
        res['name'] = name
        res['dataset'] = 2
        res['hidden'] = str(hidden)
        res['activation'] = 'relu'
        experiments.append(res)
        if res.get('test_acc_all') is not None:
            print(f"    Train: {res['train_acc']:.4f}, Test(all): {res['test_acc_all']:.4f}, "
                  f"前60: {res.get('test_acc_first60', 'N/A')}, 后150: {res.get('test_acc_last150', 'N/A')}")
        else:
            print(f"    FAILED: {res.get('error', 'unknown')}")

    # ============================================================
    # 保存所有结果
    # ============================================================
    summary_file = os.path.join(OUTPUT_DIR, 'comparison_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'实验':<20} {'名称':<22} {'TrainAcc':>10} {'TestAll':>10} {'前60':>10} {'后150':>10}\n")
        f.write("-" * 75 + "\n")
        for e in experiments:
            f.write(f"{e.get('experiment','?'):<20} {e.get('name','?'):<22} "
                    f"{e.get('train_acc','N/A'):>10} {e.get('test_acc_all','N/A'):>10} "
                    f"{e.get('test_acc_first60','N/A'):>10} {e.get('test_acc_last150','N/A'):>10}\n")

    print(f"\n\nSummary saved to: {summary_file}")

    # 保存 JSON 便于后续分析
    json_file = os.path.join(OUTPUT_DIR, 'comparison_results.json')
    json_data = []
    for e in experiments:
        d = {k: v for k, v in e.items() if k != 'output'}
        json_data.append(d)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved to: {json_file}")


if __name__ == '__main__':
    main()
