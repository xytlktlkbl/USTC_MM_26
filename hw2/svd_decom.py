import numpy as np


def vector_norm(x):
    return np.sqrt(np.sum(x * x))


def normalize(x, tol=1e-12):
    n = vector_norm(x)
    if n < tol:
        return None
    return x / n


def power_iteration_for_top_singular(A, tol=1e-10, max_iter=500, seed=None):
    """
    用幂法求矩阵 A 的最大奇异值及对应左右奇异向量

    返回:
        sigma, u, v
    其中:
        sigma: 最大奇异值
        u: 左奇异向量, shape=(m,)
        v: 右奇异向量, shape=(n,)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    v = normalize(v, tol=tol)
    if v is None:
        return 0.0, np.zeros(m), np.zeros(n)

    last_sigma = 0.0

    for _ in range(max_iter):
        # v <- A^T A v
        Av = A @ v
        sigma = vector_norm(Av)

        if sigma < tol:
            return 0.0, np.zeros(m), np.zeros(n)

        u = Av / sigma

        At_u = A.T @ u
        v_new = normalize(At_u, tol=tol)
        if v_new is None:
            return 0.0, np.zeros(m), np.zeros(n)

        # 重新估计 sigma
        Av_new = A @ v_new
        sigma_new = vector_norm(Av_new)

        if abs(sigma_new - last_sigma) < tol and vector_norm(v_new - v) < tol:
            v = v_new
            sigma = sigma_new
            u = Av_new / sigma if sigma > tol else np.zeros(m)
            return sigma, u, v

        v = v_new
        last_sigma = sigma_new

    # 达到最大迭代次数后返回当前结果
    Av = A @ v
    sigma = vector_norm(Av)
    if sigma < tol:
        return 0.0, np.zeros(m), np.zeros(n)

    u = Av / sigma
    return sigma, u, v


def top_k_svd_power(A, k, tol=1e-10, max_iter=500, seed=None):
    """
    用幂法 + deflation 求前 k 个奇异值/奇异向量

    输入:
        A: m x n
        k: 要求的前 k 个奇异值
        tol: 收敛阈值
        max_iter: 每个奇异值的最大幂法迭代次数

    返回:
        U_k: m x r
        Sigma_k: r x r
        Vt_k: r x n
        singular_values: 长度 r
    其中 r <= k（如果后面奇异值已经接近 0，会提前停止）
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    r = min(k, m, n)

    A_work = A.copy()

    U_cols = []
    V_cols = []
    singular_values = []

    rng = np.random.default_rng(seed)

    for i in range(r):
        sigma, u, v = power_iteration_for_top_singular(
            A_work,
            tol=tol,
            max_iter=max_iter,
            seed=rng.integers(0, 10**9)
        )

        if sigma < tol:
            break

        # 保存
        singular_values.append(sigma)
        U_cols.append(u)
        V_cols.append(v)

        # rank-1 deflation
        A_work = A_work - sigma * np.outer(u, v)

    if len(singular_values) == 0:
        U_k = np.zeros((m, 0))
        Sigma_k = np.zeros((0, 0))
        Vt_k = np.zeros((0, n))
        singular_values = np.array([])
        return U_k, Sigma_k, Vt_k, singular_values

    U_k = np.column_stack(U_cols)
    V_k = np.column_stack(V_cols)
    singular_values = np.array(singular_values, dtype=float)

    # 再做一次列正交化，减少数值误差
    U_k = orthonormalize_columns(U_k, tol=tol)
    V_k = orthonormalize_columns(V_k, tol=tol)

    rr = min(U_k.shape[1], V_k.shape[1], len(singular_values))
    singular_values = singular_values[:rr]
    U_k = U_k[:, :rr]
    V_k = V_k[:, :rr]

    Sigma_k = np.diag(singular_values)
    Vt_k = V_k.T

    return U_k, Sigma_k, Vt_k, singular_values


def orthonormalize_columns(X, tol=1e-10):
    """
    对矩阵列向量做 Gram-Schmidt 正交化
    """
    X = np.array(X, dtype=float)
    m, n = X.shape
    cols = []

    for j in range(n):
        v = X[:, j].copy()
        for q in cols:
            v -= np.dot(q, v) * q
        nv = vector_norm(v)
        if nv > tol:
            cols.append(v / nv)

    if len(cols) == 0:
        return np.zeros((m, 0))

    return np.column_stack(cols)


def svd_from_scratch(A, k, tol=1e-10, max_iter=500, full_matrices=False, seed=None):
    """
    基于幂法的 top-k SVD

    输入:
        A: m x n
        k: 求前 k 个奇异值
        tol: 收敛阈值
        max_iter: 每个奇异值的最大迭代次数
        full_matrices:
            False -> 返回截断版:
                U: m x r
                Sigma: r x r
                Vt: r x n
            True -> 尽量模仿完整SVD风格:
                U: m x m
                Sigma: m x n
                Vt: n x n
                但只有前 r 个奇异值有效
        seed: 随机种子

    返回:
        U, Sigma, Vt, singular_values
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    U_k, Sigma_k, Vt_k, singular_values = top_k_svd_power(
        A, k=k, tol=tol, max_iter=max_iter, seed=seed
    )

    r = len(singular_values)

    if not full_matrices:
        return U_k, Sigma_k, Vt_k, singular_values

    # full_matrices=True 时补成和原先更像的形式
    U = complete_orthonormal_basis(U_k, m, tol=tol)
    V = complete_orthonormal_basis(Vt_k.T, n, tol=tol)
    Vt = V.T

    Sigma = np.zeros((m, n))
    for i in range(r):
        Sigma[i, i] = singular_values[i]

    return U, Sigma, Vt, singular_values


def complete_orthonormal_basis(U_partial, m, tol=1e-10):
    """
    把已有列向量扩展成 m 维标准正交基
    """
    U_partial = np.array(U_partial, dtype=float)
    if U_partial.size == 0:
        U_partial = np.zeros((m, 0))

    basis = []
    for i in range(U_partial.shape[1]):
        v = U_partial[:, i].copy()
        for b in basis:
            v -= np.dot(b, v) * b
        nv = vector_norm(v)
        if nv > tol:
            basis.append(v / nv)

    for j in range(m):
        v = np.zeros(m)
        v[j] = 1.0
        for b in basis:
            v -= np.dot(b, v) * b
        nv = vector_norm(v)
        if nv > tol:
            basis.append(v / nv)
        if len(basis) == m:
            break

    return np.column_stack(basis)


def reconstruct_from_svd(U, Sigma, Vt):
    """
    根据 U, Sigma, Vt 重建矩阵
    """
    return U @ Sigma @ Vt


def reconstruct_with_k(U_k, Sigma_k, Vt_k, k=None):
    """
    对截断SVD结果再取前k项重建
    如果 U_k/Sigma_k/Vt_k 本来就是 top-k 结果，这里可直接重建
    """
    if k is None:
        return U_k @ Sigma_k @ Vt_k

    Uk = U_k[:, :k]
    Sk = Sigma_k[:k, :k]
    Vtk = Vt_k[:k, :]
    return Uk @ Sk @ Vtk


# =========================
# 示例
# =========================
if __name__ == "__main__":
    A = np.array([
        [4, 0, 2],
        [3, -5, 1],
        [0,  2, 6],
        [1,  1, 0]
    ], dtype=float)

    # 只求前2个奇异值
    U, Sigma, Vt, s = svd_from_scratch(
        A,
        k=2,
        tol=1e-8,
        max_iter=1000,
        full_matrices=False,
        seed=42
    )

    print("U =")
    print(U)
    print("\nSigma =")
    print(Sigma)
    print("\nVt =")
    print(Vt)
    print("\nsingular_values =")
    print(s)

    A_recon = reconstruct_from_svd(U, Sigma, Vt)
    print("\nA_reconstructed(top-k) =")
    print(A_recon)

    err = vector_norm(A - A_recon)
    print("\nreconstruction error =", err)