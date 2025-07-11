import argparse
import os
import random
import timeit
import warnings
import sklearn.metrics
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve, matthews_corrcoef
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from IDSAE import auto_encoder, sparse_auto_encoder, deep_sparse_auto_encoder, improved_deep_sparse_auto_encoder, \
    get_negative_sample_by_randomSample, get_negative_sample_by_KMeans, get_negative_sample_by_KMeans_and_cosine_distances, \
    draw_ROC_curve_by_five_fold, draw_ROC_curve, draw_PR_curve, draw_PR_curve_new, \
    data_toExcel, kfold_by_CV

start_time = timeit.default_timer()
warnings.filterwarnings("ignore")

# 解析命令行参数（控制交叉验证模式）
parser = argparse.ArgumentParser(description='CNISECDA')
parser.add_argument("--cv", default=3, type=int, choices=[1, 2, 3])
args = parser.parse_args()
print("dataset: CircR2Disease | cv:", args.cv)

# 设置随机种子保证可重复性
seed = 36
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# 读取circRNA和疾病的相似性融合矩阵，以及circRNA和疾病关联矩阵
CD_association_matrix = pd.read_csv("../Dataset/CircR2Disease/Association Matrix.csv", index_col=0)
circRNA_similarity_fusion_matrix = pd.read_csv('../Dataset/CircR2Disease/circRNA_similarity_fusion_matrix.csv', index_col=0)
disease_similarity_fusion_matrix = pd.read_csv('../Dataset/CircR2Disease/disease_similarity_fusion_matrix.csv', index_col=0)
CD = np.array(CD_association_matrix)
CC = np.array(circRNA_similarity_fusion_matrix)
DD = np.array(disease_similarity_fusion_matrix)

# 读取ChebNet_CNN处理过的特征矩阵cnn_outputs_dataframe
cnn_outputs = pd.read_csv("cnn_outputs_dataframe.csv", index_col=0)
cnn_outputs = np.array(cnn_outputs)

# 根据cnn_outputs构建特征向量
circ_nums = CD.shape[0]
dis_nums = CD.shape[1]
features_embedding_circ = cnn_outputs[0:circ_nums, :]
features_embedding_dis = cnn_outputs[circ_nums:cnn_outputs.shape[0], :]

# 正样本特征拼接（circRNA特征 + 疾病特征）
positive_index_tuple = np.where(CD == 1)
positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))
all_train_features_input = []
all_train_lable = []
for (r, d) in positive_index_list:
    all_train_features_input.append(np.hstack((features_embedding_circ[r, :], features_embedding_dis[d, :])))
    all_train_lable.append(1)

# 根据特征矩阵生成负样本集
negative_index_tuple = np.where(CD == 0)
negative_index_list = list(zip(negative_index_tuple[0], negative_index_tuple[1]))
NEGATIVE_SAMPLE_CHA_ALL = []
for (r, d) in negative_index_list:
    NEGATIVE_SAMPLE_CHA_ALL.append(np.hstack((features_embedding_circ[r, :], features_embedding_dis[d, :])))
NEGATIVE_SAMPLE_CHA_ALL = np.array(NEGATIVE_SAMPLE_CHA_ALL)

# 使用KMeans聚类方法选择最佳的负样本
NEGATIVE_SAMPLE_CHA, NEGATIVE_SAMPLE_CHA_LABEL = get_negative_sample_by_KMeans_and_cosine_distances(
    NEGATIVE_SAMPLE_CHA_ALL,
    len(positive_index_list))

# 特征编码与标准化
all_train_features_input = np.array(all_train_features_input)
all_features_input = np.vstack((all_train_features_input, NEGATIVE_SAMPLE_CHA))
all_label = np.array(all_train_lable).reshape(-1, 1)
all_label = np.vstack((all_label, NEGATIVE_SAMPLE_CHA_LABEL))

# 稀疏自编码器：降低特征维度，提取关键信息。
CHA_data = improved_deep_sparse_auto_encoder(all_features_input)
# 原始特征和编码后特征的维度对比
print("原始特征维度:", all_features_input.shape)
print("编码后特征维度:", CHA_data.shape)  # 原始代码中CHA_data是编码后的结果

# 定义交叉验证划分（非标准KFold）
row = 105
col = 17
cv = args.cv
train_index_all, test_index_all = kfold_by_CV(CHA_data, 5, row, col, cv)
# kfold = KFold(n_splits=5, shuffle=True, random_state=36)

all_auc = []
all_aupr = []
all_accuracy = []
all_precision = []
all_recall = []
all_f1 = []
all_mcc = []
# 用于绘制ROC、PR曲线的参数列表
FPR = []
TPR = []
PRECISION = []
RECALL = []
test_label_all = []
test_predict_prob_all = []

# 模型训练与评估
for i in range(5):
    train_features_input, train_label = CHA_data[train_index_all[i]], all_label[train_index_all[i]]
    test_features_input, test_label = CHA_data[test_index_all[i]], all_label[test_index_all[i]]

    scaler = StandardScaler()
    train_features_input = scaler.fit_transform(train_features_input)
    test_features_input = scaler.transform(test_features_input)

    # CatBoost
    catboost = CatBoostClassifier(iterations=20, learning_rate=0.001, depth=4, verbose=0, random_seed=36)

    # train_label.ravel() 将训练标签展平为一维数组
    catboost.fit(train_features_input, train_label.ravel())
    test_predict = catboost.predict(test_features_input)
    test_predict_prob = catboost.predict_proba(test_features_input)[:, 1]

    # 计算评估指标
    accuracy = accuracy_score(test_label, test_predict)
    precision = precision_score(test_label, test_predict, average='macro')
    recall = recall_score(test_label, test_predict, average='macro')
    f1 = f1_score(test_label, test_predict, average='macro')
    mcc = matthews_corrcoef(test_label, test_predict)

    # 计算auc、aupr和对应的绘制曲线
    auc = roc_auc_score(test_label, test_predict_prob)
    fpr, tpr, thresholds1 = roc_curve(test_label, test_predict_prob, pos_label=1)
    pre, rec, thresholds2 = precision_recall_curve(test_label, test_predict_prob, pos_label=1)
    aupr = sklearn.metrics.auc(rec, pre)

    FPR.append(fpr)
    TPR.append(tpr)
    PRECISION.append(pre)
    RECALL.append(rec)
    test_label_all.append(test_label)
    test_predict_prob_all.append(test_predict_prob)

    print("auc:{}".format(auc))
    print("aupr:{}".format(aupr))
    print("accuracy:{}".format(accuracy))
    print("precision:{}".format(precision))
    print("recall:{}".format(recall))
    print("f1_score:{}".format(f1))
    print("mcc:{}".format(mcc))
    all_auc.append(auc)
    all_aupr.append(aupr)
    all_accuracy.append(accuracy)
    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_mcc.append(mcc)
    # # 选择将fpr、tpr（pre、rec）写入Excel表格
    # filepath = "./Result/cv%d/" % (cv)
    # os.makedirs(filepath, exist_ok=True)
    # data_toExcel(fpr, tpr, filepath + "AUC_CV%d_randomSample_%.4f.xlsx" % (cv, auc), "HMDAD_CV%d_AUC" % cv)
    # data_toExcel(rec, pre, filepath + "AUPR_CV%d_randomSample_%.4f.xlsx" % (cv, aupr), "HMDAD_CV%d_AUPR" % cv)

# 汇总并打印评估结果
mean_auc = np.around(np.mean(np.array(all_auc)), 4)
mean_aupr = np.around(np.mean(np.array(all_aupr)), 4)
mean_accuracy = np.around(np.mean(np.array(all_accuracy)), 4)
mean_precision = np.around(np.mean(np.array(all_precision)), 4)
mean_recall = np.around(np.mean(np.array(all_recall)), 4)
mean_f1 = np.around(np.mean(np.array(all_f1)), 4)
mean_mcc = np.around(np.mean(np.array(all_mcc)), 4)

# 计算标准差
std_auc = np.around(np.std(np.array(all_auc)), 4)
std_aupr = np.around(np.std(np.array(all_aupr)), 4)
std_accuracy = np.around(np.std(np.array(all_accuracy)), 4)
std_precision = np.around(np.std(np.array(all_precision)), 4)
std_recall = np.around(np.std(np.array(all_recall)), 4)
std_f1 = np.around(np.std(np.array(all_f1)), 4)
std_mcc = np.around(np.std(np.array(all_mcc)), 4)

print()
print("MEAN AUC:{} ± {}".format(mean_auc, std_auc))
print("MEAN AUPR:{} ± {}".format(mean_aupr, std_aupr))
print("MEAN ACCURACY:{} ± {}".format(mean_accuracy, std_accuracy))
print("MEAN PRECISION:{} ± {}".format(mean_precision, std_precision))
print("MEAN RECALL:{} ± {}".format(mean_recall, std_recall))
print("MEAN F1_SCORE:{} ± {}".format(mean_f1, std_f1))
print("MEAN MCC:{} ± {}".format(mean_mcc, std_mcc))

# 计算运行时间
end_time = timeit.default_timer()
print("Running time: %s Seconds" % (end_time - start_time))

# # 绘制ROC、PR曲线
# draw_ROC_curve(FPR, TPR, cv)
# draw_ROC_curve_by_five_fold(FPR, TPR)
#
# draw_PR_curve(test_label_all, test_predict_prob_all, cv)
# draw_PR_curve_new(RECALL, PRECISION)
