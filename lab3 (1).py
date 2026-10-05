"""
LAB 3: TRỊ RIÊNG & VECTOR RIÊNG
Gộp toàn bộ 5 bài vào một file. Không dùng numpy.
Chạy: python lab3_all.py
"""

import math


# =====================================================================
# BÀI 1: Xác thực Trị riêng & Vector riêng (A * x == lambda * x)
# =====================================================================
def verify_eigen(A, x, lambda_val, eps=1e-6):
    n = len(A)
    # Bước 1: vế trái LHS = A * x
    Ax = []
    for i in range(n):
        Ax.append(sum(A[i][j] * x[j] for j in range(len(x))))
    # Bước 2: vế phải RHS = lambda * x
    lambda_x = [lambda_val * val for val in x]
    # Bước 3 + 4: so sánh sai số
    ok = all(abs(Ax[i] - lambda_x[i]) < eps for i in range(n))
    return ok, Ax, lambda_x


def bai1():
    print("=" * 60)
    print("BÀI 1: Xác thực Trị riêng & Vector riêng")
    print("=" * 60)
    A = [[4, 2], [1, 3]]
    x = [2, 1]
    lambda_val = 5

    is_valid, Ax, lambda_x = verify_eigen(A, x, lambda_val)
    print(f"Ax: {Ax} | Lambda*x: {lambda_x}")
    print(f"x là vector riêng: {is_valid}")

    # Thử thêm trường hợp sai để kiểm chứng hàm
    is_valid2, Ax2, lx2 = verify_eigen(A, x, 4)
    print(f"\nThử lambda = 4 -> Ax: {Ax2} | Lambda*x: {lx2}")
    print(f"x là vector riêng: {is_valid2}")


# =====================================================================
# BÀI 2: Lũy thừa nhanh ma trận bằng chéo hóa A^k = P * D^k * P^(-1)
# =====================================================================
def mat_mul_2x2(X, Y):
    """Nhân hai ma trận 2x2."""
    return [
        [sum(X[i][t] * Y[t][j] for t in range(2)) for j in range(2)]
        for i in range(2)
    ]


def matrix_power_fast(P, D_diag, P_inv, k):
    # Bước 1 + 2: D^k là ma trận đường chéo với các phần tử d_i ** k
    D_k_diag = [val ** k for val in D_diag]
    D_k = [[D_k_diag[0], 0.0], [0.0, D_k_diag[1]]]
    # Bước 3: P * D^k * P^(-1)
    return mat_mul_2x2(mat_mul_2x2(P, D_k), P_inv)


def matrix_power_naive(A, k):
    """Nhân ma trận lặp k lần (dùng để đối chứng)."""
    R = [[1.0, 0.0], [0.0, 1.0]]
    for _ in range(k):
        R = mat_mul_2x2(R, A)
    return R


def bai2():
    print("\n" + "=" * 60)
    print("BÀI 2: Lũy thừa nhanh ma trận bằng chéo hóa")
    print("=" * 60)
    # A = [[4, 2], [1, 3]] có trị riêng 5 (vector [2,1]) và 2 (vector [1,-1])
    A = [[4, 2], [1, 3]]
    D_diag = [5, 2]
    P = [[2.0, 1.0], [1.0, -1.0]]              # các cột là vector riêng
    P_inv = [[1 / 3, 1 / 3], [1 / 3, -2 / 3]]  # nghịch đảo của P

    print("P * D * P^-1 (k=1):",
          [[round(v, 6) for v in row] for row in matrix_power_fast(P, D_diag, P_inv, 1)])

    for k in (2, 5, 10):
        fast = matrix_power_fast(P, D_diag, P_inv, k)
        slow = matrix_power_naive(A, k)
        print(f"\nk = {k}")
        print("  Chéo hóa :", [[round(v, 4) for v in row] for row in fast])
        print("  Nhân lặp :", [[round(v, 4) for v in row] for row in slow])


# =====================================================================
# BÀI 3: Trừ trung bình & Ma trận hiệp phương sai
# =====================================================================
def mean_centering(X):
    """Trừ trung bình từng cột. X: M hàng, N cột."""
    M = len(X)
    N = len(X[0])
    means = [sum(X[k][j] for k in range(M)) / M for j in range(N)]
    return [[X[k][j] - means[j] for j in range(N)] for k in range(M)]


def compute_covariance_matrix(X_centered):
    """Cov[i][j] = sum_k Xc[k][i] * Xc[k][j] / (M - 1)."""
    M = len(X_centered)
    N = len(X_centered[0])
    cov = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            cov[i][j] = sum(X_centered[k][i] * X_centered[k][j] for k in range(M)) / (M - 1)
    return cov


def bai3():
    print("\n" + "=" * 60)
    print("BÀI 3: Trừ trung bình & Ma trận hiệp phương sai")
    print("=" * 60)
    X = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
    ]
    Xc = mean_centering(X)
    print("X đã trừ trung bình:")
    for row in Xc:
        print([round(v, 4) for v in row])

    cov = compute_covariance_matrix(Xc)
    print("\nMa trận hiệp phương sai:")
    for row in cov:
        print([round(v, 4) for v in row])

    # Kiểm tra kích thước bất kỳ (M=4, N=3)
    Y = [[1, 2, 3], [4, 5, 6], [7, 8, 10], [2, 1, 0]]
    cov_y = compute_covariance_matrix(mean_centering(Y))
    print("\nCov 3x3:")
    for row in cov_y:
        print([round(v, 4) for v in row])


# =====================================================================
# BÀI 4: Chiếu dữ liệu lên Thành phần chính
# =====================================================================
def project_data_1d(X_centered, pc_vector):
    """Tích vô hướng của mỗi hàng với pc_vector -> M tọa độ mới."""
    return [
        sum(row[j] * pc_vector[j] for j in range(len(pc_vector)))
        for row in X_centered
    ]


def bai4():
    print("\n" + "=" * 60)
    print("BÀI 4: Chiếu dữ liệu lên Thành phần chính")
    print("=" * 60)
    X_centered = [
        [0.69, 0.49],
        [-1.31, -1.21],
        [0.39, 0.99],
        [0.09, 0.29],
        [1.29, 1.09],
    ]
    pc = [0.7071, 0.7071]  # vector chuẩn hóa (độ dài ~ 1)
    result = project_data_1d(X_centered, pc)
    print("Tọa độ sau khi chiếu:", [round(v, 4) for v in result])


# =====================================================================
# BÀI 5: Pipeline giảm chiều dữ liệu bệnh án & Explained Variance
# =====================================================================
medical_data = [
    [120, 95, 210, 24.5],
    [140, 130, 250, 29.0],
    [110, 85, 180, 21.5],
    [155, 160, 280, 32.0],
    [130, 105, 220, 26.0],
]


def jacobi_eigen(S, tol=1e-12, max_iter=100):
    """Trị riêng / vector riêng của ma trận đối xứng S bằng thuật toán Jacobi.
    Trả về (trị riêng giảm dần, danh sách vector riêng tương ứng)."""
    n = len(S)
    A = [row[:] for row in S]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        # Tìm phần tử ngoài đường chéo lớn nhất
        p, q, big = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > big:
                    big, p, q = abs(A[i][j]), i, j
        if big < tol:
            break
        theta = (A[q][q] - A[p][p]) / (2 * A[p][q])
        t = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta ** 2 + 1))
        c = 1 / math.sqrt(t ** 2 + 1)
        s = t * c
        # Quay Givens: A <- J^T A J, V <- V J
        for k in range(n):
            akp, akq = A[k][p], A[k][q]
            A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
        for k in range(n):
            apk, aqk = A[p][k], A[q][k]
            A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq

    eigvals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigvals[i], reverse=True)
    vals = [eigvals[i] for i in order]
    vecs = [[V[k][i] for k in range(n)] for i in order]  # vecs[i] = vector riêng i
    # Quy ước dấu: thành phần lớn nhất (theo trị tuyệt đối) của mỗi vector là dương
    for v in vecs:
        m = max(v, key=abs)
        if m < 0:
            for j in range(len(v)):
                v[j] = -v[j]
    return vals, vecs


def pca_reduce_2d(data):
    Xc = mean_centering(data)                 # 1. Trừ trung bình
    cov = compute_covariance_matrix(Xc)       # 2. Ma trận hiệp phương sai 4x4
    eigvals, eigvecs = jacobi_eigen(cov)      # 3. Trị riêng / vector riêng
    pc1, pc2 = eigvecs[0], eigvecs[1]         # 4. Hai thành phần chính lớn nhất
    z1 = project_data_1d(Xc, pc1)             # 5. Chiếu dữ liệu
    z2 = project_data_1d(Xc, pc2)
    reduced = [[z1[i], z2[i]] for i in range(len(data))]
    return reduced, eigvals, (pc1, pc2)


def explained_variance_ratio(lambdas, k=2):
    return sum(lambdas[:k]) / sum(lambdas)


def bai5():
    print("\n" + "=" * 60)
    print("BÀI 5: Pipeline PCA dữ liệu bệnh án & Explained Variance")
    print("=" * 60)
    reduced, eigvals, (pc1, pc2) = pca_reduce_2d(medical_data)

    print("Trị riêng tính từ dữ liệu:", [round(v, 4) for v in eigvals])
    print("PC1:", [round(v, 4) for v in pc1])
    print("PC2:", [round(v, 4) for v in pc2])
    print("\nDữ liệu sau khi giảm về 2 chiều (PC1, PC2):")
    for i, r in enumerate(reduced, 1):
        print(f"  Bệnh nhân {i}: ({r[0]:8.4f}, {r[1]:8.4f})")
    print(f"\nTỷ lệ giải thích (tính từ dữ liệu): {explained_variance_ratio(eigvals):.4%}")

    # Theo đề bài: lambda = [145.2, 32.8, 4.5, 1.2]
    lambdas = [145.2, 32.8, 4.5, 1.2]
    ratio = (lambdas[0] + lambdas[1]) / sum(lambdas)
    print(f"Ratio theo đề (lambda cho sẵn): {ratio:.4f} ({ratio * 100:.2f}%)")

    # ---------- Phân tích ý nghĩa học máy ----------
    # Nếu tỷ lệ thông tin giữ lại > 90% (ở đây ~96.9%), việc loại bỏ 2 chiều cuối
    # KHÔNG làm mất bản chất dữ liệu: PC1 và PC2 đã chứa gần như toàn bộ phương
    # sai (tức là sự khác biệt giữa các bệnh nhân). Hai chiều bị bỏ có phương sai
    # rất nhỏ (~3%), phần lớn là nhiễu hoặc thông tin dư thừa vì các chỉ số như
    # huyết áp, đường huyết, cholesterol, BMI vốn tương quan mạnh với nhau.
    # Lợi ích khi trực quan hóa 2D cho chuyên gia y tế: nhìn được cả tập bệnh
    # nhân trên một biểu đồ phân tán, dễ phát hiện nhóm bệnh nhân nguy cơ cao,
    # bệnh nhân bất thường (outlier) và xu hướng chung mà không phải đọc 4 con số
    # cho từng người; ngoài ra còn giảm chi phí tính toán/lưu trữ và giảm nguy
    # cơ overfitting khi huấn luyện mô hình. Lưu ý: PC là tổ hợp tuyến tính của
    # các chỉ số gốc nên cần xem hệ số của PC1, PC2 để diễn giải ý nghĩa y khoa.


if __name__ == "__main__":
    bai1()
    bai2()
    bai3()
    bai4()
    bai5()
