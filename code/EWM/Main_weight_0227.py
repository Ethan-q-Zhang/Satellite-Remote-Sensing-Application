import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


class CityWeightModel:
    def __init__(self, data):
        self.data = data
        self.years = data.columns
        self.n_indicators = data.shape[0]

    # ===============================
    # 1️⃣ 熵权法
    # ===============================
    # def entropy_weight(self):
    #     X = self.data.values.astype(float)
    #
    #     scaler = MinMaxScaler()
    #     X_norm = scaler.fit_transform(X.T).T
    #     X_norm = np.where(X_norm == 0, 1e-6, X_norm)
    #
    #     P = X_norm / X_norm.sum(axis=0, keepdims=True)
    #     k = 1 / np.log(self.n_indicators)
    #     E = -k * np.sum(P * np.log(P), axis=1)
    #
    #     d = 1 - E
    #     w = d / d.sum()
    #
    #     self.initial_weights = w
    #     return w

    def entropy_weight(self):
        data_norm = self.data
        # 计算比重
        p = data_norm.div(data_norm.sum(axis=1), axis=0)

        # 计算熵值
        k = 1 / np.log(self.n_indicators)
        p = p.astype('float64')
        e = -k * (p * np.log(p)).sum(axis=1)
        e = e.replace([np.inf, -np.inf, np.nan], 0)  # 处理异常值
        # 计算权重
        d = 1 - e
        w = d / d.sum()
        self.initial_weights = w
        return w

    # ===============================
    # 2️⃣ MLP优化熵权法权重
    # ===============================
    def ml_adjustment(self, alpha=0.15):
        X = []
        y = []
        data_matrix = self.data.values

        for t in range(1, data_matrix.shape[1]):
            prev_year = data_matrix[:, t - 1]
            current_year = data_matrix[:, t]
            X.append(prev_year)
            y.append(current_year - prev_year)

        X = np.array(X)
        y = np.array(y)

        model = MLPRegressor(hidden_layer_sizes=(12, 6),   #(20, 10)
                             activation='relu',
                             max_iter=2000,
                             random_state=42)

        model.fit(X, y)

        last_data = data_matrix[:, -1].reshape(1, -1)   #data_matrix[:, -1]用最后一个（或倒数第二个）年份输入产生delta_pred
        delta_pred = model.predict(last_data)[0]
        delta_pred = np.tanh(delta_pred)

        new_weights = self.initial_weights * (1 + alpha * delta_pred)
        # new_weights = self.initial_weights + alpha * delta_pred
        new_weights = np.clip(new_weights, 0, None)
        new_weights = new_weights / new_weights.sum()

        self.optimized_weights = new_weights
        self.mlp_model = model

        return new_weights

    # ===============================
    # 3️⃣ 预测2024年权重
    # ===============================
    def predict_2024_weights(self, alpha=0.15):
        data_matrix = self.data.values
        last_data = data_matrix[:, -1].reshape(1, -1)

        delta_pred = self.mlp_model.predict(last_data)[0]  #使用已经训练好的 self.mlp_model直接预测
        delta_pred = np.tanh(delta_pred)

        future_weights = self.optimized_weights * (1 + alpha * delta_pred)
        future_weights = np.clip(future_weights, 0, None)
        future_weights = future_weights / future_weights.sum()

        return future_weights


def run_city(city_name, group1_df, group2_df):
    results = {}

    for group_name, df in zip(["Group1", "Group2"], [group1_df, group2_df]):

        model = CityWeightModel(df)

        w_entropy = model.entropy_weight()
        w_optimized = model.ml_adjustment()
        w_2024 = model.predict_2024_weights()

        result_df = pd.DataFrame({
            "Indicator": df.index,
            "Entropy_Weight": w_entropy,
            "Optimized_Weight": w_optimized,
            "Predicted_2024_Weight": w_2024
        })

        results[group_name] = result_df

    return results


# ==========================================
# 主程序（多城市，多sheet）
# ==========================================
if __name__ == "__main__":

    input_file = "城市数据熵权v4.xlsx"
    output_file = "All_Cities_Weight_Results_12_6_2000_2014-2022×1.xlsx"

    # 读取所有sheet名称
    xls = pd.ExcelFile(input_file)
    sheet_names = xls.sheet_names

    with pd.ExcelWriter(output_file) as writer:

        for city_name in sheet_names:

            print(f"正在处理城市：{city_name}")

            df = pd.read_excel(input_file,
                               sheet_name=city_name,
                               index_col=0)

            # 假设前10行是Group1，后10行是Group2
            group1_df = df.iloc[0:10, 0:9]
            group2_df = df.iloc[11:21, 0:9]

            results = run_city(city_name, group1_df, group2_df)

            # 输出到Excel，每个城市两个sheet
            for group_name, result_df in results.items():
                sheet_out_name = f"{city_name}_{group_name}"
                result_df.to_excel(writer, sheet_name=sheet_out_name, index=False)

    print("全部城市计算完成，结果已输出。")