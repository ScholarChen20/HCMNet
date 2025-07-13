import os
from abc import ABC

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
# 设置随机种子确保可复现性
np.random.seed(42)
torch.manual_seed(42)


# ======================== 数据加载与预处理函数 ========================
def load_and_preprocess_data(dataset_path):
    """加载数据集并进行预处理"""
    # 加载数据
    data = pd.read_csv(dataset_path)

    # 分离特征和标签（最后一列为标签）
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values


    y = y-y.min() # 标签从0开始
    y = y.astype(int) # 标签类型转换为int

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, scaler


# ======================== 改进的LDA实现 ========================
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder


class ImprovedLDA:
    def __init__(self, n_components=None, reg_param=1.0):
        """
        正则化LDA：对类内散度矩阵加入L2正则
        :param n_components: 投影后维度（多分类时≤n_classes-1）
        :param reg_param: 正则化系数λ
        """
        self.n_components = n_components
        self.reg_param = reg_param
        self.means = None  # 每类的均值 (n_classes, n_features)
        self.Sw_proj = None  # 投影空间的类内散度矩阵 (n_components, n_components)
        self.Sw_proj_inv = None  # 投影空间的类内散度矩阵逆
        self.projection = None  # 投影矩阵 (n_features, n_components)
        self.label_encoder = LabelEncoder()  # 标签编码
        self.means_proj = None  # 投影后的类均值

    def fit(self, X, y):
        """
        训练LDA：计算均值、散度矩阵、投影矩阵
        :param X: 特征数组 (n_samples, n_features)
        :param y: 标签数组 (n_samples,)
        """
        # 确保输入为NumPy数组
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        y_enc = self.label_encoder.fit_transform(y)  # 标签转为0-based
        n_classes = len(self.label_encoder.classes_)
        n_features = X.shape[1]

        print(f"训练LDA: 样本数={X.shape[0]}, 特征数={n_features}, 类别数={n_classes}")

        # 1. 计算每类均值
        self.means = np.zeros((n_classes, n_features))
        for c in range(n_classes):
            mask = (y_enc == c)
            self.means[c] = X[mask].mean(axis=0)

        # 2. 计算类内散度矩阵Sw（带正则化）
        Sw = np.zeros((n_features, n_features))
        for c in range(n_classes):
            mask = (y_enc == c)
            X_c = X[mask] - self.means[c]
            Sw_c = X_c.T @ X_c
            Sw += Sw_c

        # 加入L2正则：Sw = Sw + λ*I
        Sw_reg = Sw + self.reg_param * np.eye(n_features)

        # 3. 计算类间散度矩阵Sb
        global_mean = X.mean(axis=0)
        Sb = np.zeros((n_features, n_features))
        for c in range(n_classes):
            n_c = (y_enc == c).sum()
            mean_diff = self.means[c] - global_mean
            Sb += n_c * np.outer(mean_diff, mean_diff)

        # 4. 求解广义特征值问题：Sb w = λ Sw w
        # 转换为标准特征值问题：Sw^{-1} Sb w = λ w
        Sw_inv = np.linalg.pinv(Sw_reg)  # 使用伪逆提高数值稳定性

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(Sw_inv @ Sb)
        eigenvalues = eigenvalues.real  # 取实部
        eigenvectors = eigenvectors.real

        # 按特征值从大到小排序，选择前n_components个
        sorted_indices = np.argsort(eigenvalues)[::-1]  # 降序排列
        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        self.projection = eigenvectors[:, sorted_indices[:self.n_components]]

        # 5. 计算投影后的类均值
        self.means_proj = self.means @ self.projection

        # 6. 计算投影空间的类内散度矩阵及其逆
        # 投影后的Sw: W^T * Sw * W
        self.Sw_proj = self.projection.T @ Sw_reg @ self.projection
        self.Sw_proj_inv = np.linalg.inv(self.Sw_proj + 1e-6 * np.eye(self.n_components))  # 加入小值正则防止奇异

        print(f"投影矩阵形状: {self.projection.shape}, 投影均值形状: {self.means_proj.shape}")

    def transform(self, X):
        """
        特征投影
        :param X: 特征数组 (n_samples, n_features)
        :return: 投影后特征 (n_samples, n_components)
        """
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return X @ self.projection

    def predict(self, X):
        """
        分类预测：投影后计算到各类均值的马氏距离，取最小距离类
        :param X: 特征数组 (n_samples, n_features)
        :return: 预测标签 (n_samples,)
        """
        # 将输入转换为NumPy数组
        if isinstance(X, torch.Tensor):
            X = X.numpy()

        # 特征投影
        X_proj = self.transform(X)

        # 初始化距离矩阵
        n_samples = X_proj.shape[0]
        n_classes = self.means_proj.shape[0]
        distances = np.zeros((n_samples, n_classes))

        # 计算每个样本到各类均值的马氏距离
        for i in range(n_samples):
            for c in range(n_classes):
                diff = X_proj[i] - self.means_proj[c]
                # 马氏距离: (x - μ)^T Σ^{-1} (x - μ)
                dist = diff @ self.Sw_proj_inv @ diff.T
                distances[i, c] = dist

        # 预测最小距离的类别
        pred_indices = np.argmin(distances, axis=1)
        return self.label_encoder.inverse_transform(pred_indices)


# ======================== 改进的MLP实现（含AFIM模块） ========================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层→隐藏层
        self.relu = nn.ReLU()                        # 激活函数
        self.fc2 = nn.Linear(hidden_dim, output_dim) # 隐藏层→输出层

    def forward(self, x):
        x = self.fc1(x)   # 线性变换
        x = self.relu(x)  # 激活
        x = self.fc2(x)   # 输出层变换
        return x

#  ========================= 改进的MLP实现=======================
class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, adaptive_depth=True, dropout_rate=0.5):
        """
        增强型MLP，适用于多种数据集

        参数:
        input_dim: 输入特征维度
        output_dim: 输出类别数
        hidden_dim: 基础隐藏层维度 (默认512)
        adaptive_depth: 是否根据输入维度自动调整网络深度 (默认True)
        dropout_rate: dropout比例 (默认0.5)
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # 自适应网络深度：根据输入维度调整隐藏层数量
        if adaptive_depth:
            # 计算网络深度，基于输入特征维度
            self.depth = max(2, min(5, int(input_dim ** 0.5 // 3)))
            print(f"自适应网络深度: {self.depth}层 (基于输入维度{input_dim})")
        else:
            self.depth = 3  # 默认3层隐藏层

        # 创建网络层
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.LeakyReLU(negative_slope=0.01))
        self.layers.append(nn.Dropout(dropout_rate))

        # 隐藏层 - 根据计算深度创建
        for i in range(self.depth - 1):
            # 后续层维度递减
            next_dim = max(128, hidden_dim // (2 ** (i + 1)))
            self.layers.append(nn.Linear(hidden_dim, next_dim))
            self.layers.append(nn.BatchNorm1d(next_dim))
            self.layers.append(nn.LeakyReLU(negative_slope=0.01))
            self.layers.append(nn.Dropout(dropout_rate))
            hidden_dim = next_dim  # 更新下一层的输入维度

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """自定义权重初始化"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # 使用Kaiming初始化，针对LeakyReLU优化
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)

        # 输出层使用更小的权重初始化
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.01)

    def forward(self, x):
        """前向传播"""
        # 遍历所有层
        for layer in self.layers:
            x = layer(x)

        # 输出层
        x = self.output_layer(x)
        return x

    def get_config(self):
        """返回模型配置信息"""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate
        }

class LeNet5(nn.Module, ABC):
    """
    略微修改过的 LeNet5 模型
    Attributes:
        need_dropout (bool): 是否需要增加随机失活层
        conv1 (nn.Conv2d): 卷积核1，默认维度 (6, 5, 5)
        pool1 (nn.MaxPool2d): 下采样函数1，维度 (2, 2)
        conv2 (nn.Conv2d): 卷积核2，默认维度 (16, 5, 5)
        pool2 (nn.MaxPool2d): 下采样函数2，维度 (2, 2)
        conv3 (nn.Conv2d): 卷积核3，默认维度 (120, 5, 5)
        fc1 (nn.Linear): 全连接函数1，维度 (120, 84)
        fc2 (nn.Linear): 全连接函数2，维度 (84, 10)
        dropout (nn.Dropout): 随机失活函数
    """
    def __init__(self, dropout_prob=0., halve_conv_kernels=False):
        """
        初始化模型各层函数
        :param dropout_prob: 随机失活参数
        :param halve_conv_kernels: 是否将卷积核数量减半
        """
        super(LeNet5, self).__init__()
        kernel_nums = [6, 16]
        if halve_conv_kernels:
            kernel_nums = [num // 2 for num in kernel_nums]
        self.need_dropout = dropout_prob > 0

        # 卷积层 1，6个 5*5 的卷积核
        # 由于输入图像是 28*28，所以增加 padding=2，扩充到 32*32
        self.conv1 = nn.Conv2d(1, kernel_nums[0], (5, 5), padding=2)
        # 下采样层 1，采样区为 2*2
        self.pool1 = nn.MaxPool2d((2, 2))
        # 卷积层 2，16个 5*5 的卷积核
        self.conv2 = nn.Conv2d(kernel_nums[0], kernel_nums[1], (5, 5))
        # 下采样层 2，采样区为 2*2
        self.pool2 = nn.MaxPool2d((2, 2))
        # 卷积层 3，120个 5*5 的卷积核
        self.conv3 = nn.Conv2d(kernel_nums[1], 120, (5, 5))
        # 全连接层 1，120*84 的全连接矩阵
        self.fc1 = nn.Linear(120, 84)
        # 全连接层 2，84*10 的全连接矩阵
        self.fc2 = nn.Linear(84, 10)
        # 随机失活层，失活率为 dropout_prob
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        前向传播函数，返回给定输入数据的预测标签数组
        :param x: 维度为 (batch_size, 28, 28) 的图像数据
        :return: 维度为 (batch_size, 10) 的预测标签
        """
        x = x.view(-1, 1, 28, 28) # (batch_size, 1, 28, 28)
        feature_map = self.conv1(x)             # (batch_size, 6, 28, 28)
        feature_map = self.pool1(feature_map)   # (batch_size, 6, 14, 14)
        feature_map = self.conv2(feature_map)   # (batch_size, 16, 10, 10)
        feature_map = self.pool2(feature_map)   # (batch_size, 16, 5, 5)
        feature_map = self.conv3(feature_map).squeeze()     # (batch_size, 120)
        out = self.fc1(feature_map)             # (batch_size, 84)
        if self.need_dropout:
            out = self.dropout(out)             # (batch_size, 10)
        out = self.fc2(out)                     # (batch_size, 10)
        return out

# ======================== 模型评估函数 ========================
def evaluate_model(y_true, y_pred, dataset_name, model_name):
    average_method = 'binary' if len(np.unique(y_true)) == 2 else 'macro'  #  二分类问题使用二分类评价指标，多分类问题使用宏平均
    """评估模型性能并输出结果"""
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average=average_method)
    pre = precision_score(y_true, y_pred, average=average_method)
    f1 = f1_score(y_true, y_pred, average=average_method)

    print(f"\n===== {dataset_name} - {model_name} =====")
    print(f"准确率 (ACC): {acc:.4f}")
    print(f"召回率 (REC): {rec:.4f}")
    print(f"精确率 (PRE): {pre:.4f}")
    print(f"F1值: {f1:.4f}")

    # 绘制混淆矩阵
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title(f'{dataset_name} - {model_name}-Confusion Matrix')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('Truth Label')
    # save_dir = 'results/pr_confusion_matrices'
    # os.makedirs(save_dir, exist_ok=True)
    # plt.savefig(os.path.join(save_dir, f'{dataset_name}_{model_name}_confusion.png'))
    # plt.close()

    return acc, rec, pre, f1


# ======================== 主训练函数 ========================
def train_and_evaluate(dataset_path, dataset_name):
    """在单个数据集上训练和评估所有模型"""
    # 加载和预处理数据
    X, y, scaler = load_and_preprocess_data(dataset_path)
    results = {}
    validation_accuracies = {
        'SVM': [],  # 存储LDA的准确率（每个fold一个值）
        'MLP': []
    }  # 用于存储每个模型的验证准确率

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== {dataset_name} 数据集 - 第 {fold + 1}/5 折 ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # # 训练和评估改进的LDA
        # ilda = LinearDiscriminantAnalysis()  # 原始LDA
        # # ilda = ImprovedLDA(reg_param=0.5)  # 改进的LDA
        # ilda.fit(X_train, y_train)
        # y_pred = ilda.predict(X_test)
        # ilda_metrics = evaluate_model(y_test, y_pred, dataset_name, "LDA")
        # # 保存LDA的准确率
        # val_acc_lda = accuracy_score(y_test, y_pred)
        # validation_accuracies['LDA'].append(val_acc_lda)
        # ✅ 新增：SVM 分类器
        svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # RBF核函数
        svm_clf.fit(X_train, y_train)
        y_pred_svm = svm_clf.predict(X_test)
        svm_metrics = evaluate_model(y_test, y_pred_svm, dataset_name, "SVM")
        val_acc_svm = accuracy_score(y_test, y_pred_svm)
        validation_accuracies['SVM'].append(val_acc_svm)

        # 训练和评估改进的MLP
        # 转换数据为PyTorch张量
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        # 创建数据加载器
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 初始化模型
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y))
        # print(f"输入维度: {input_dim}, 输出维度: {output_dim}")  #输入维度8，输出维度2
        mlp = MLP(input_dim, 320, output_dim)  # 原始MLP
        # mlp = EnhancedMLP(input_dim, output_dim)  #改进后的MLP

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # optimizer = optim.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=0.0001)

        # 验证准确率列表
        val_accs = []
        # 训练模型
        mlp.train()
        for epoch in range(100):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                # print(batch_x.shape, batch_y.shape)  #64,8 64,1
                optimizer.zero_grad()
                outputs = mlp(batch_x)
                # print(outputs.shape)  #64,2
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # 验证准确率
            mlp.eval()
            with torch.no_grad():
                outputs = mlp(X_test_t)
                _, y_pred = torch.max(outputs, 1)
                val_acc = accuracy_score(y_test, y_pred.numpy())
                val_accs.append(val_acc)
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/100, Loss: {total_loss / len(train_loader):.4f}")
        # 保存验证准确率
        validation_accuracies['MLP'] = val_accs

        # 评估模型
        mlp.eval()
        with torch.no_grad():
            outputs = mlp(X_test_t)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.numpy()

        mlp_metrics = evaluate_model(y_test, y_pred, dataset_name, "MLP")
        # 存储结果
        results[fold] = {
            'SVM': svm_metrics,
            'ImprovedMLP': mlp_metrics
        }


    # 计算平均性能
    avg_results = {
        'SVM': np.mean([results[f]['SVM'] for f in results], axis=0),
        'ImprovedMLP': np.mean([results[f]['ImprovedMLP'] for f in results], axis=0)
    }

    print(f"\n===== {dataset_name} 数据集平均性能 =====")
    for model, metrics in avg_results.items():
        acc, rec, pre, f1 = metrics
        print(f"{model} - ACC: {acc:.4f}, REC: {rec:.4f}, PRE: {pre:.4f}, F1: {f1:.4f}")

     # 绘制验证准确率曲线
    # plt.figure(figsize=(10, 6))
    # for model, val_accs in validation_accuracies.items():
    #     if model == 'LDA':
    #         continue
    #     plt.plot(range(1, len(val_accs) + 1), val_accs, label=model)
    # lda_avg_acc = float(np.mean(validation_accuracies['LDA']))
    # plt.axhline(y=lda_avg_acc, color='r', linestyle='--', linewidth=2, label='LDA (avg ACC)')
    # plt.title(f'{dataset_name} - Validation ACC Comparison')
    # plt.xlabel('Epoch')
    # plt.ylabel('Validation Accuracy')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    # save_dir = 'results/pr_validation_curves'
    # os.makedirs(save_dir, exist_ok=True)
    # plt.savefig(os.path.join(save_dir, f'{dataset_name}_validation_accuracy.png'))
    # plt.close()
    # plt.show()

    # 在全局变量中记录验证准确率曲线
    global all_validation_curves

    # LDA 的验证准确率只有一个值（因为只评估一次）
    all_validation_curves[f"{dataset_name}-SVM"] = np.mean(validation_accuracies['SVM'])

    # MLP 的是 list，每个 epoch 都有
    all_validation_curves[f"{dataset_name}-MLP"] = validation_accuracies['MLP']

    return avg_results

def plot_all_validation_curves(all_validation_curves, num_epochs=100):
    plt.figure(figsize=(12, 6))

    epochs = list(range(1, num_epochs + 1))

    for key in all_validation_curves:
        if 'MLP' in key:  # 只画 MLP 的曲线（LDA 是标量）
            curve = all_validation_curves[key]
            plt.plot(epochs, curve, label=key)
        # else:
        #     acc = all_validation_curves[key]
        #     plt.axhline(y=acc, color=np.random.choice(['r', 'g', 'b', 'c']), linestyle='--', label=f'{key} (LDA)')

    plt.title('Validation Accuracy Curves Across Datasets and Models')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/validation_curve_comparison.png')
    plt.show()

# ======================== 主执行流程 ========================
if __name__ == "__main__":
    # 数据集路径
    datasets = {
        "HTRU_2": "data/HTRU_2.csv",
        "letter": "data/letter.csv",
        "msplice": "data/msplice.csv",
        "spambase": "data/spambase.csv"
    }

    all_results = {}
    all_validation_curves = {} # 用于存储每个模型的验证准确率

    # 遍历所有数据集
    for name, path in datasets.items():
        print(f"\n{'=' * 50}")
        print(f"开始处理数据集: {name}")
        print(f"{'=' * 50}")
        results = train_and_evaluate(path, name)
        all_results[name] = results

    # 打印最终结果汇总
    print("\n\n===== 所有数据集最终结果汇总 =====")
    for dataset, models in all_results.items():
        print(f"\n{dataset}:")
        for model, metrics in models.items():
            acc, rec, pre, f1 = metrics
            print(f"  {model}: ACC={acc:.4f}, F1={f1:.4f}")

    # === 新增绘图逻辑 ===
    plot_all_validation_curves(all_validation_curves)
    # 结果可视化
    # models = ['ImprovedLDA', 'ImprovedMLP']
    # metrics = ['ACC', 'F1']
    #
    # # 创建子图（1行，metrics数量列）
    # fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    #
    # for i, metric in enumerate(metrics):
    #     ax = axes[i]
    #     width = 0.35
    #     dataset_names = list(datasets.keys())  # 提取数据集名列表（如 ["HTRU_2", "letter", ...]）
    #     x = np.arange(len(dataset_names))  # x轴位置，对应每个数据集
    #
    #     for j, model in enumerate(models):
    #         # 对每个模型，收集所有数据集的该指标值
    #         values = []
    #         for dataset in dataset_names:
    #             if metric == 'ACC':
    #                 # 假设 evaluate_model 返回 (acc, rec, pre, f1)，ACC对应第0位
    #                 val = all_results[dataset][model][0]
    #             else:  # F1 对应第3位
    #                 val = all_results[dataset][model][3]
    #             values.append(val)
    #
    #         # 绘制分组柱状图（每个模型占一个分组）
    #         ax.bar(x + j * width, values, width, label=model)
    #
    #     # 子图样式配置
    #     ax.set_title(f'{metric} Comparison')
    #     ax.set_xlabel('Datasets')
    #     ax.set_ylabel(metric)
    #     ax.set_xticks(x + width / 2)  # 调整x轴刻度到分组中间
    #     ax.set_xticklabels(dataset_names)  # 用数据集名作为x轴标签
    #     ax.legend()  # 显示图例
    #     ax.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    #
    # plt.tight_layout()  # 自动调整子图间距，避免重叠
    # plt.savefig('dataset_comparison.png')  # 保存可视化结果
    # plt.show()  # 显示图像