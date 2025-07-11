import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
基于CircR2Disease数据库整合微生物和疾病的各种相似度
'''


# 计算circRNA的功能相似性
def circRNA_functional_similarity():
    # 修改目录即可
    # circRNA、疾病关联矩阵
    CD = pd.read_excel("../Dataset/CircR2Disease/Association Matrixs.xlsx", sheet_name="Association Matrix", header=None)
    # 疾病语义相似性矩阵
    DS = pd.read_excel("../Dataset/CircR2Disease/disease_semantic_similarity.xlsx", header=None)
    # 将DataFrame类型转为numpy类型，等价于DS = np.array(DS.values)，舍弃了索引名和列名
    DS = np.array(DS)
    C_num = CD.shape[0]
    T = []
    for i in range(C_num):
        T.append(np.where(CD.iloc[i] == 1))
    FS = []
    for i in range(C_num):
        for j in range(C_num):
            T0_T1 = []
            if len(T[i][0]) != 0 and len(T[j][0]) != 0:
                for ti in T[i][0]:
                    max_ = []
                    for tj in T[j][0]:
                        max_.append(DS[ti][tj])
                    T0_T1.append(max(max_))
            if len(T[i][0]) == 0 or len(T[j][0]) == 0:
                T0_T1.append(0)
            T1_T0 = []
            if len(T[i][0]) != 0 and len(T[j][0]) != 0:
                for tj in T[j][0]:
                    max_ = []
                    for ti in T[i][0]:
                        max_.append(DS[tj][ti])
                    T1_T0.append(max(max_))
            if len(T[i][0]) == 0 or len(T[j][0]) == 0:
                T1_T0.append(0)

            a = len(T[i][0])
            b = len(T[j][0])
            S1 = sum(T0_T1)
            S2 = sum(T1_T0)
            fs = []
            if a != 0 and b != 0:
                fsim = (S1 + S2) / (a + b)
                fs.append(fsim)
            if a == 0 or b == 0:
                fs.append(0)
            FS.append(fs)
    FS = np.array(FS).reshape(C_num, C_num)
    FS = pd.DataFrame(FS)
    # 下列代码有点冗余，因为原本矩阵对角线上的值就全是1
    for index, rows in FS.iterrows():
        for col, rows in FS.iterrows():
            if index == col:
                FS.loc[index, col] = 1
    # 修改数据框的索引名和列名
    FS.index = CD.index
    FS.columns = CD.index
    return FS


# 根据关联矩阵计算cirnRNA和疾病的GIP相似性
def GIP_similarity():
    CD = pd.read_excel("../Dataset/CircR2Disease/Association Matrixs.xlsx", sheet_name="Association Matrix", header=None)
    # 首先求出circRNA和disease的带宽参数rm，rd
    circRNA_num = CD.shape[0]
    disease_num = CD.shape[1]
    EUC_C = np.linalg.norm(CD, ord=2, axis=1, keepdims=False)
    EUC_D = np.linalg.norm(CD.T, ord=2, axis=1, keepdims=False)
    SUM_EUC_C = np.sum(EUC_C ** 2)
    SUM_EUC_D = np.sum(EUC_D ** 2)
    rm = 1 / ((1 / circRNA_num) * SUM_EUC_C)
    rd = 1 / ((1 / disease_num) * SUM_EUC_D)
    # 计算circRNA和disease的GIP
    circRNA_GIP = pd.DataFrame(0, index=CD.index, columns=CD.index)
    disease_GIP = pd.DataFrame(0, index=CD.columns, columns=CD.columns)
    CD = np.mat(CD)
    for i in range(circRNA_num):
        for j in range(circRNA_num):
            c_norm = np.linalg.norm(CD[i] - CD[j], ord=2, axis=1, keepdims=False)
            c_norm = c_norm ** 2
            c_norm_result = np.exp(-rm * c_norm)
            circRNA_GIP.iloc[i, j] = c_norm_result
    for i in range(disease_num):
        for j in range(disease_num):
            d_norm = np.linalg.norm(CD.T[i] - CD.T[j], ord=2, axis=1, keepdims=False)
            d_norm = d_norm ** 2
            d_norm_result = np.exp(-rd * d_norm)
            disease_GIP.iloc[i, j] = d_norm_result
    return circRNA_GIP, disease_GIP


if __name__ == '__main__':
    circRNA_functional_similarity = circRNA_functional_similarity()
    circRNA_GIP_similarity, disease_GIP_similarity = GIP_similarity()
    circRNA_functional_similarity.to_csv("../Dataset/CircR2Disease/circRNA_functional_similarity.csv")
    circRNA_GIP_similarity.to_csv("../Dataset/CircR2Disease/circRNA_GIP_similarity.csv")
    disease_GIP_similarity.to_csv("../Dataset/CircR2Disease/disease_GIP_similarity.csv")

    circRNA_similarity_fusion_matrix = pd.DataFrame(0, index=circRNA_functional_similarity.index,
                                                    columns=circRNA_functional_similarity.index)
    circRNA_functional_similarity = np.array(circRNA_functional_similarity)
    circRNA_GIP_similarity = np.array(circRNA_GIP_similarity)
    for i in range(circRNA_functional_similarity.shape[0]):
        for j in range(circRNA_functional_similarity.shape[1]):
            if circRNA_functional_similarity[i, j] != 0:
                circRNA_similarity_fusion_matrix.iloc[i, j] = (circRNA_functional_similarity[i, j] +
                                                               circRNA_GIP_similarity[i, j]) / 2
            else:
                circRNA_similarity_fusion_matrix.iloc[i, j] = circRNA_GIP_similarity[i, j]

    circRNA_similarity_fusion_matrix.to_csv("../Dataset/CircR2Disease/circRNA_similarity_fusion_matrix.csv")

    disease_do_similarity = pd.read_excel("../Dataset/CircR2Disease/disease_semantic_similarity.xlsx", header=None)
    disease_similarity_fusion_matrix = pd.DataFrame(0, index=disease_do_similarity.index,
                                                    columns=disease_do_similarity.index)
    disease_do_similarity = np.array(disease_do_similarity)
    disease_GIP_similarity = np.array(disease_GIP_similarity)
    for i in range(disease_do_similarity.shape[0]):
        for j in range(disease_do_similarity.shape[1]):
            if disease_do_similarity[i, j] != 0:
                disease_similarity_fusion_matrix.iloc[i, j] = (disease_do_similarity[i, j] +
                                                               disease_GIP_similarity[i, j]) / 2
            else:
                disease_similarity_fusion_matrix.iloc[i, j] = disease_GIP_similarity[i, j]
    disease_similarity_fusion_matrix.to_csv("../Dataset/CircR2Disease/disease_similarity_fusion_matrix.csv")

