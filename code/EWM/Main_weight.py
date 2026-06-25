import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from numpy.random import dirichlet
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class AdaptiveWeightFramework:
    def __init__(self, data):
        """
        初始化自适应权重计算框架

        参数:
            data: DataFrame，行是指标，列是样本/年份
        """
        self.data = data.copy()
        self.indicators = data.index.values
        self.samples = data.columns.values
        self.initial_weights = self._calculate_entropy_weights()

    def _calculate_entropy_weights(self):
        """计算熵权法初始权重"""
        # 标准化处理
        # data_norm = self.data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)
        data_norm = self.data
        # 计算比重
        p = data_norm.div(data_norm.sum(axis=1), axis=0)

        # 计算熵值
        k = 1 / np.log(len(self.samples))
        p = p.astype('float64')
        e = -k * (p * np.log(p)).sum(axis=1)
        e = e.replace([np.inf, -np.inf, np.nan], 0)  # 处理异常值

        # 计算权重
        d = 1 - e
        w = d / d.sum()

        return w.values

    def least_squares_adjustment(self, lambda_param=0.5):
        """
        最小二乘自适应权重调整
        参数:
            lambda_param: 正则化参数，控制权重调整幅度
        返回:
            调整后的权重数组
        """
        X = self.data.values.T[:-1]  # 使用前n-1个样本作为训练数据,X是设计矩阵（特征矩阵），每行代表一个样本，每列代表一个特征。
        y = np.ones(len(X))  # 假设综合得分为1

        # 定义目标函数
        def objective(w):
            pred = X.dot(w)    # 计算线性预测值,即模型对目标值 y 的预测
            return np.sum((y - pred) ** 2) + lambda_param * np.sum((w - self.initial_weights) ** 2)
            # np.sum((y - pred) ** 2):计算预测值 pred 与真实值 y 之间的均方误差（MSE）
            # lambda_param * np.sum((w - self.initial_weights) ** 2):L2正则化项(也称为岭回归或Tikhonov正则化)，用于约束权重向量 w 不要偏离初始权重太远

        # 定义约束条件
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # 通过字典定义等式约束，确保权重向量 w 的所有元素之和为1。
        bounds = [(0, 1) for _ in range(len(self.initial_weights))]     # 限制每个权重 wi 的取值范围为 [0, 1]

        # 优化求解
        result = minimize(objective, self.initial_weights, method='SLSQP',  # method='SLSQP'：序列最小二乘规划算法，适用于非线性目标函数+等式/不等式约束的问题
                          bounds=bounds, constraints=constraints)  # minimize 进行带约束的优化，核心目标是在权重和为1且每个权重位于[0,1]区间的约束下，最小化目标函数


        return result.x

    def bayesian_adjustment(self, n_samples=1000):
        """
        贝叶斯动态权重调整
        参数:
            n_samples: 采样次数
        返回:
            调整后的权重数组
        """
        # 计算各指标的变化趋势作为伪计数
        trends = self.data.diff(axis=1).mean(axis=1).values
        alpha = self.initial_weights * 100 + trends * 20  # 调整超参数

        # 从Dirichlet分布中采样
        alpha = alpha.astype('float64')
        samples = dirichlet(alpha, size=n_samples)
        new_weights = samples.mean(axis=0)

        return new_weights

    def ml_adjustment(self, alpha=0.14):
        """
        基于机器学习的自适应权重调整alpha
        参数:
            alpha: 学习率，控制调整幅度
        返回:
            调整后的权重数组
        """
        X = []
        y = []

        # 构建时间序列特征
        for i in range(1, len(self.samples)):
            prev_data = self.data.iloc[:, i - 1].values
            current_data = self.data.iloc[:, i].values
            X.append(prev_data)
            y.append(current_data - prev_data)  # 预测变化量

        X = np.array(X)
        y = np.array(y)

        # 训练神经网络模型
        model = MLPRegressor(hidden_layer_sizes=(20, 10), activation='relu',  #20,10
                             max_iter=1000, random_state=42)
        model.fit(X, y)

        # 预测下一个样本的变化
        last_data = self.data.iloc[:, -2].values.reshape(1, -1)
        delta_pred = model.predict(last_data)[0]

        # 结合初始权重进行调整
        new_weights = self.initial_weights + alpha * delta_pred
        new_weights = np.clip(new_weights, 0, 1)  # 限制在0-1之间
        new_weights = new_weights / new_weights.sum()  # 归一化

        return new_weights

    def calculate_all_methods(self):
        """计算所有方法的权重"""
        return {
            'initial': self.initial_weights,
            'least_squares': self.least_squares_adjustment(),
            'bayesian': self.bayesian_adjustment(),
            'ml_model': self.ml_adjustment()
        }

    def visualize_results(self, weights_dict=None):
        """可视化权重比较结果"""
        if weights_dict is None:
            weights_dict = self.calculate_all_methods()

        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.indicators))
        width = 0.2

        for i, (method, weights) in enumerate(weights_dict.items()):
            plt.bar(x + (i - 1.5) * width, weights, width, label=method)

        plt.xlabel('指标')
        plt.ylabel('权重')
        plt.title('不同方法的权重调整结果')
        plt.xticks(x, self.indicators, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()


# 测试用例
if __name__ == "__main__":
    # 准备测试数据（行是指标，列是年份）
    # test_data = {
    #     '2015': [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0001, 1.0001, 0.0001, 0.0001, 1.0001],
    #     '2016': [0.1086, 0.0496, 0.1751, 0.0915, 0.1112, 0.611, 0.6853, 0.152, 0.1086, 0.9506],
    #     '2017': [0.2908, 0.1578, 0.2431, 0.3858, 0.5557, 0.637, 0.6297, 0.2469, 0.2908, 0.8424],
    #     '2018': [0.4793, 0.3088, 0.1972, 0.5982, 0.6668, 0.6392, 0.352, 0.3988, 0.4793, 0.6914],
    #     '2019': [0.76, 0.468, 0.1384, 0.7207, 0.6668, 0.6361, 0.3705, 0.576, 0.76, 0.5322],
    #     '2020': [0.7848, 0.9796, 0.3172, 1.0001, 0.6668, 0.0001, 0.1853, 0.6014, 0.7848, 0.0206],
    #     '2021': [1.0001, 1.0001, 1.0001, 0.9254, 1.0001, 0.2675, 0.0001, 1.0001, 1.0001, 0.0001]
    # }
    #
    # indicators = [
    #     'GDP/亿元', '人口/万人', '财政收入', '资产投资额', '三产业比',
    #     '消费品零', '耕地比例', '设用地比', '济密度 亿', '密度 人/'
    # ]
    # df = pd.DataFrame(test_data, index=indicators)

    df = pd.read_excel('D:\study\Revisepaper\Paperdata\城市数据熵权.xlsx', sheet_name='珠海')
    data = df.iloc[0:10, 0:11]   # les 对应的数据
    # data = df.iloc[11:20, 0:8]    # res 对应的数据
    indicators = data['年份']  # 年份数据
    years = data.columns[1:].values  # 排除'年份'列后的指标名称
    test_data = data.values[:, 1:]

    # 创建DataFrame（行是指标，列是样本）
    df = pd.DataFrame(test_data, index=indicators, columns=years)  # index 参数对应 DataFrame 的行标签，columns 对应列标签。


    # 创建计算框架实例
    framework = AdaptiveWeightFramework(df)

    # 计算并打印各种方法的权重
    results = framework.calculate_all_methods()

    matrix = []
    for method, weights in results.items():
        print(f"{method} weights:", dict(zip(indicators, weights)))
        matrix.append(weights)
    matrix = pd.DataFrame(matrix)   # 打印四种方案的权重initial weights、least_squares weights、bayesian weights、ml_model weights
    print(matrix.T)
    # # 可视化结果
    # framework.visualize_results()
