########################################
#   在Main_weight.py的基础上，仅进行循环计算优化
#   不改变熵权法与MLP权重调整算法
########################################

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from numpy.random import dirichlet
import matplotlib.pyplot as plt

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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

        # 保持原有逻辑：此处默认输入数据已经完成标准化
        data_norm = self.data

        # 计算比重
        p = data_norm.div(data_norm.sum(axis=1), axis=0)

        # 计算熵值
        k = 1 / np.log(len(self.samples))
        p = p.astype('float64')
        e = -k * (p * np.log(p)).sum(axis=1)
        e = e.replace([np.inf, -np.inf, np.nan], 0)

        # 计算权重
        d = 1 - e
        w = d / d.sum()

        return w.values

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
            y.append(current_data - prev_data)

        X = np.array(X)
        y = np.array(y)

        # 训练神经网络模型
        model = MLPRegressor(
            hidden_layer_sizes=(20, 10),
            activation='relu',
            max_iter=1000,
            random_state=42
        )
        model.fit(X, y)

        # 预测下一个样本的变化
        last_data = self.data.iloc[:, -2].values.reshape(1, -1)
        delta_pred = model.predict(last_data)[0]

        # 结合初始权重进行调整
        new_weights = self.initial_weights + alpha * delta_pred
        new_weights = np.clip(new_weights, 0, 1)
        new_weights = new_weights / new_weights.sum()   # 权重归一化

        return new_weights

    def calculate_all_methods(self):
        """计算所有方法的权重"""
        return {
            'initial': self.initial_weights,
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


def calculate_one_group(city_name, group_name, group_df):
    """
    计算单个城市、单个指标组的权重。

    参数:
        city_name: 城市名称
        group_name: Group1 或 Group2
        group_df: 行为指标、列为年份的数据

    返回:
        result_df: 当前城市当前组的计算结果
    """

    # 转为数值型，避免 Excel 中混入文本导致计算异常
    group_df = group_df.apply(pd.to_numeric, errors='coerce')

    # 若存在全空行或全空列，删除
    group_df = group_df.dropna(axis=0, how='all')
    group_df = group_df.dropna(axis=1, how='all')

    # 若存在个别空值，用 0 填充，保持原算法可运行
    group_df = group_df.fillna(0)

    framework = AdaptiveWeightFramework(group_df)
    results = framework.calculate_all_methods()

    result_df = pd.DataFrame({
        "City": city_name,
        "Group": group_name,
        "Indicator": group_df.index,
        "Entropy_Weight": results["initial"],
        "MLP_Optimized_Weight": results["ml_model"]
    })

    return result_df


def run_city(city_name, group1_df, group2_df):
    """
    对单个城市的 Group1 和 Group2 循环计算。

    不改变原有流程：
        熵权法初始权重 -> MLP优化权重
    """
    results = {}

    for group_name, group_df in zip(
        ["Group1", "Group2"],
        [group1_df, group2_df]
    ):
        result_df = calculate_one_group(city_name, group_name, group_df)
        results[group_name] = result_df

    return results


def safe_sheet_name(sheet_name):
    """
    Excel sheet 名称最长为31个字符，且不能包含部分特殊字符。
    """
    invalid_chars = ['\\', '/', '*', '[', ']', ':', '?']
    for ch in invalid_chars:
        sheet_name = sheet_name.replace(ch, "_")

    return sheet_name[:31]


# ==========================================
# 主程序：多城市、多sheet循环计算
# ==========================================
if __name__ == "__main__":

    input_file = r"D:\study\Revisepaper\Paperdata\城市数据熵权.xlsx"
    output_file = r"D:\study\Revisepaper\Paperdata\All_Cities_Weight_Results_0509_20_10纵向.xlsx"

    xls = pd.ExcelFile(input_file)
    sheet_names = xls.sheet_names

    all_summary = []

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        for city_name in sheet_names:
            print(f"正在处理城市：{city_name}")

            # 读取当前城市 sheet
            # 第一列作为指标名称，后续列作为年份/样本
            df = pd.read_excel(
                input_file,
                sheet_name=city_name,
                index_col=0
            )

            # ==============================
            # 根据你的数据结构划分指标组
            # ==============================
            # Group1：前10个指标
            group1_df = df.iloc[0:10, 0:10]

            # Group2：第12到第21行，共10个指标
            # 如果你的第11行是空行或分隔行，则使用 11:21
            group2_df = df.iloc[11:21, 0:10]

            # 如果你的 Group2 紧接 Group1，没有空行，则改成：
            # group2_df = df.iloc[10:20, 0:10]

            results = run_city(city_name, group1_df, group2_df)

            for group_name, result_df in results.items():
                sheet_out_name = safe_sheet_name(f"{city_name}_{group_name}")
                result_df.to_excel(writer, sheet_name=sheet_out_name, index=False)

                all_summary.append(result_df)

        # # 汇总结果 纵向
        # summary_df = pd.concat(all_summary, axis=0, ignore_index=True)
        # summary_df.to_excel(writer, sheet_name="All_Summary", index=False)

        # ==============================
        # All_Summary：各城市横向排列
        # ==============================
        summary_blocks = []

        for result_df in all_summary:
            city_name = result_df["City"].iloc[0]
            group_name = result_df["Group"].iloc[0]

            block = result_df[[
                "Indicator",
                "Entropy_Weight",
                "MLP_Optimized_Weight"
            ]].copy()

            # 给每个城市和分组增加前缀，避免列名重复
            block.columns = [
                f"{city_name}_{group_name}_Indicator",
                f"{city_name}_{group_name}_Entropy_Weight",
                f"{city_name}_{group_name}_MLP_Optimized_Weight"
            ]

            summary_blocks.append(block.reset_index(drop=True))

        summary_wide_df = pd.concat(summary_blocks, axis=1)
        summary_wide_df.to_excel(writer, sheet_name="All_Summary", index=False)

    print("全部城市计算完成，结果已输出：")
    print(output_file)