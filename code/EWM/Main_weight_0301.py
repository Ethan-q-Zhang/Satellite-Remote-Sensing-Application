import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

class CityWeightModel:
    def __init__(self, data: pd.DataFrame):
        """
        data: DataFrame (指标 × 年份)
              行：指标（10）
              列：年份（2014-2023等）
        """
        self.data = data.copy()
        self.years = list(data.columns)
        self.n_indicators = data.shape[0]

        self.initial_weights = None
        self.optimized_weights = None
        self.mlp_model = None

    # ===============================
    # 1️⃣ 熵权法（按行：每个指标一个权重）
    # ===============================
    def entropy_weight(self, data: pd.DataFrame | None = None) -> pd.Series:
        """
        返回：Series，index=指标，value=权重
        """
        if data is None:
            data = self.data

        data_norm = data.astype(float)

        # 计算比重：每个指标在各年份的占比（按行归一）
        p = data_norm.div(data_norm.sum(axis=1).replace(0, np.nan), axis=0)
        p = p.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 避免log(0)
        p = p.replace(0, 1e-12).astype("float64")

        # 计算熵值：对年份求和 -> 每个指标一个熵
        k = 1 / np.log(data.shape[1])  # 年份数
        e = -k * (p * np.log(p)).sum(axis=1)

        # 计算权重
        d = 1 - e
        w = d / d.sum()

        return w

    # ===============================
    # 2️⃣ MLP优化熵权法权重（仅用于权重优化，不改变你原流程）
    # ===============================
    def ml_adjustment(self, alpha) -> pd.Series:
        """
        训练MLP学习“指标变化量”，然后用最后一个（或倒数第二个）年份输入产生delta_pred，
        在熵权权重基础上进行“比例扰动”得到优化权重。
        返回：Series，index=指标，value=优化权重
        """
        # 初始权重（2014-2023）
        self.initial_weights = self.entropy_weight(self.data)

        X = []
        y = []
        data_matrix = self.data.values.astype(float)  # (指标, 年份)

        # 构建时间序列训练对：x_t -> delta_{t+1}
        for t in range(1, data_matrix.shape[1]):
            prev_year = data_matrix[:, t - 1]
            current_year = data_matrix[:, t]
            X.append(prev_year)
            y.append(current_year - prev_year)

        X = np.array(X)  # (年份-1, 指标)
        y = np.array(y)  # (年份-1, 指标)

        model = MLPRegressor(
            hidden_layer_sizes=(20, 10), #12，6
            activation="relu",
            max_iter=1000,
            random_state=42
        )
        model.fit(X, y)

        # ⚠ 保留你原设定：用倒数第二列作为输入（你注释里写了 data_matrix[:, -1]）
        last_data = data_matrix[:, -1].reshape(1, -1)
        delta_pred = model.predict(last_data)[0]
        delta_pred = np.tanh(delta_pred)

        ##########################################
        #     权重优化模型选择，加和承               #
        ##########################################
        # new_weights = self.initial_weights.values * (1 + alpha * delta_pred)
        new_weights = self.initial_weights + alpha * delta_pred
        new_weights = np.clip(new_weights, 0, None)
        new_weights = new_weights / new_weights.sum()

        self.optimized_weights = pd.Series(new_weights, index=self.data.index, name="OptimizedWeight")
        self.mlp_model = model

        return self.optimized_weights

    # ===============================
    # 3️⃣ 预测2024年权重（保留你原流程：用已训练的 self.mlp_model）
    # ===============================
    def predict_2024_weights(self, alpha) -> pd.Series:
        """
        返回：Series，index=指标，value=预测2024权重
        """
        if self.mlp_model is None or self.optimized_weights is None:
            raise RuntimeError("请先调用 ml_adjustment() 训练模型并得到 optimized_weights。")

        data_matrix = self.data.values.astype(float)
        last_data = data_matrix[:, -1].reshape(1, -1)  # 用2023输入预测下一步变化

        delta_pred = self.mlp_model.predict(last_data)[0]
        delta_pred = np.tanh(delta_pred)

        future_weights = self.optimized_weights.values * (1 + alpha * delta_pred)
        # future_weights = self.optimized_weights.values + alpha * delta_pred
        future_weights = np.clip(future_weights, 0, None)
        future_weights = future_weights / future_weights.sum()

        return pd.Series(future_weights, index=self.data.index, name="Predicted2024Weight")

    # ============================================================
    # ✅ 新增 A：预测2024年各指标值（不是权重）
    # ============================================================
    def predict_2024_indicator_values(self) -> pd.Series:
        """
        用已训练的 self.mlp_model 预测 2024 指标变化量，并由 x_2023 + delta 得到 x_2024
        返回：Series，index=指标，value=预测的2024指标值
        """
        if self.mlp_model is None:
            raise RuntimeError("请先调用 ml_adjustment() 训练模型。")

        data_matrix = self.data.values.astype(float)
        x_2023 = data_matrix[:, -1]  # 2023指标向量（归一化后，理论在[0,1]）
        delta_2024 = self.mlp_model.predict(x_2023.reshape(1, -1))[0]

        # # ✅ 限制变化量幅度，避免一步跳太大（可显著减少越界概率）
        delta_2024 = np.tanh(delta_2024)

        x_2024_pred = x_2023 + delta_2024
        # 若你希望约束为非负（可选）：x_2024_pred = np.clip(x_2024_pred, 0, None)
        # ✅ 关键：归一化指标的可行域约束
        # x_2024_pred = np.clip(x_2024_pred, 0.0, 1.0)
        x_2024_pred = np.clip(x_2024_pred, 0.0, None)
        return pd.Series(x_2024_pred, index=self.data.index, name="Predicted2024Value")

    # ============================================================
    # ✅ 新增 B：将2014-2024（含预测）重新计算熵权与MLP优化
    # ============================================================
    def recompute_weights_with_2024(self, group_name, alpha_opt) -> tuple[pd.Series, pd.Series]:
        """
        基于扩展数据(2014-2024)：先熵权，再用MLP（同样的差分训练逻辑）做权重优化
        返回：
          w_entropy_2014_2024, w_optimized_2014_2024
        """
        # 1) 预测2024指标值
        x_2024 = self.predict_2024_indicator_values()

        # 2) 组成扩展数据 2014-2024
        year_2024 = "2024"
        data_ext = self.data.copy()
        # 如果列是int年份，保持一致
        if len(self.years) > 0 and isinstance(self.years[-1], (int, np.integer)):
            year_2024 = 2024
        data_ext[year_2024] = x_2024.values
        if group_name == "Group1":
            # Group1 特定的归一化策略
            for row in data_ext.index:
                if row == '人口密度 人/km2':
                    data_ext.loc[row, :] = (data_ext.loc[row, :].max() - data_ext.loc[row, :]) / (
                            data_ext.loc[row, :].max() - data_ext.loc[row, :].min())
                else:  # 其他行正向归一化
                    data_ext.loc[row, :] = (data_ext.loc[row, :] - data_ext.loc[row, :].min()) / (
                            data_ext.loc[row, :].max() - data_ext.loc[row, :].min())
        elif group_name == "Group2":
            # Group2 特定的归一化策略
            for row in data_ext.index:
                if row == '工业废水排放量（万吨）' or row == '二氧化硫排放量（吨）':  # 第 5 行和第 6 行反向归一化
                    data_ext.loc[row, :] = (data_ext.loc[row, :].max() - data_ext.loc[row, :]) / (
                            data_ext.loc[row, :].max() - data_ext.loc[row, :].min())
                else:  # 其他行正向归一化
                    data_ext.loc[row, :] = (data_ext.loc[row, :] - data_ext.loc[row, :].min()) / (
                            data_ext.loc[row, :].max() - data_ext.loc[row, :].min())


        # 3) 重新计算熵权（基于2014-2024）
        w_entropy_ext = self.entropy_weight(data_ext)

        # 4) 用扩展数据训练差分MLP，并优化权重（不影响你原流程，只是“追加结果”）
        X = []
        y = []
        data_matrix = data_ext.values.astype(float)
        for t in range(1, data_matrix.shape[1]):
            prev_year = data_matrix[:, t - 1]
            current_year = data_matrix[:, t]
            X.append(prev_year)
            y.append(current_year - prev_year)

        X = np.array(X)
        y = np.array(y)

        model_ext = MLPRegressor(
            hidden_layer_sizes=(12, 6),
            activation="relu",
            max_iter=2000,
            random_state=42
        )
        model_ext.fit(X, y)

        # 用2024前一年（即2023）作为输入，产生“末端趋势”来优化扩展熵权
        last_data_ext = data_matrix[:, -2].reshape(1, -1)  # 仍用倒数第二列（2023）
        delta_pred_ext = np.tanh(model_ext.predict(last_data_ext)[0])

        w_opt_ext = w_entropy_ext.values * (1 + alpha_opt * delta_pred_ext)
        # w_opt_ext = w_entropy_ext.values + alpha_opt * delta_pred_ext
        w_opt_ext = np.clip(w_opt_ext, 0, None)
        w_opt_ext = w_opt_ext / w_opt_ext.sum()

        w_opt_ext = pd.Series(w_opt_ext, index=data_ext.index, name="OptimizedWeight_2014_2024")
        w_entropy_ext.name = "EntropyWeight_2014_2024"

        return w_entropy_ext, w_opt_ext

def run_city(city_name: str, group1_df: pd.DataFrame, group2_df: pd.DataFrame):
    """
    不改变原有流程：熵权 -> MLP优化 -> 预测2024年权重
    然后追加：
      - 预测2024指标值
      - 基于2014-2024重新算熵权与优化权重
    """
    results = {}
    for group_name, df in zip(["Group1", "Group2"], [group1_df, group2_df]):
        model = CityWeightModel(df)

        # ===== 原流程 =====
        w_entropy = model.entropy_weight()  # 2014-2023 熵权
        model.initial_weights = w_entropy
        w_optimized = model.ml_adjustment(alpha=0.14)  # 2014-2023 MLP优化权重（并训练mlp_model）
        w_2024_weight = model.predict_2024_weights(alpha=0.14)  # 预测2024年权重

        result_df = pd.DataFrame({
            "Indicator": df.index,
            "Entropy_Weight_2014_2023": w_entropy.values,
            "Optimized_Weight_2014_2023": w_optimized.values,
            "Predicted_2024_Weight": w_2024_weight.values
        })

        # ===== 追加需求 A：预测2024指标值 =====
        pred_2024_values = model.predict_2024_indicator_values()
        result_df["Predicted_2024_IndicatorValue"] = pred_2024_values.values

        # ===== 追加需求 B：基于2014-2024重新计算与优化权重 =====
        w_entropy_ext, w_opt_ext = model.recompute_weights_with_2024(group_name, alpha_opt=0.14)
        result_df["Entropy_Weight_2014_2024"] = w_entropy_ext.values
        result_df["Optimized_Weight_2014_2024"] = w_opt_ext.values

        results[group_name] = result_df

    return results


#     # ===============================
#     # **** 预测2023年权重（由2014-2022年数据计算-开始）****
#     # ===============================
#     def predict_2023_weights(self, alpha) -> pd.Series:
#         """
#         返回：Series，index=指标，value=预测2024权重
#         """
#         if self.mlp_model is None or self.optimized_weights is None:
#             raise RuntimeError("请先调用 ml_adjustment() 训练模型并得到 optimized_weights。")
#
#         data_matrix = self.data.values.astype(float)
#         last_data = data_matrix[:, -1].reshape(1, -1)  # 用2023输入预测下一步变化
#
#         delta_pred = self.mlp_model.predict(last_data)[0]
#         delta_pred = np.tanh(delta_pred)
#
#         future_weights = self.optimized_weights.values * (1 + alpha * delta_pred)
#         # future_weights = self.optimized_weights.values + alpha * delta_pred
#         future_weights = np.clip(future_weights, 0, None)
#         future_weights = future_weights / future_weights.sum()
#
#         return pd.Series(future_weights, index=self.data.index, name="Predicted2023Weight")
#
#         # ============================================================
#         # ✅ 新增 A：预测2023年各指标值（不是权重）
#         # ============================================================
#     def predict_2023_indicator_values(self) -> pd.Series:
#         """
#         用已训练的 self.mlp_model 预测 2023 指标变化量，并由 x_2022 + delta 得到 x_2023
#         返回：Series，index=指标，value=预测的2024指标值
#         """
#         if self.mlp_model is None:
#             raise RuntimeError("请先调用 ml_adjustment() 训练模型。")
#
#         data_matrix = self.data.values.astype(float)
#         x_2022 = data_matrix[:, -1]  # 2023指标向量（归一化后，理论在[0,1]）
#         delta_2023 = self.mlp_model.predict(x_2022.reshape(1, -1))[0]
#
#         # # ✅ 限制变化量幅度，避免一步跳太大（可显著减少越界概率）
#         delta_2023 = np.tanh(delta_2023)
#
#         x_2023_pred = x_2022 + delta_2023
#         # 若你希望约束为非负（可选）：x_2023_pred = np.clip(x_2023_pred, 0, None)
#         # ✅ 关键：归一化指标的可行域约束
#         # x_2023_pred = np.clip(x_2023_pred, 0.0, 1.0)
#         x_2023_pred = np.clip(x_2023_pred, 0.0, None)
#         return pd.Series(x_2023_pred, index=self.data.index, name="Predicted2023Value")
#
#         # ============================================================
#         # ✅ 新增 C：预测2023年各指标的未归一化值
#         # ============================================================
#     def predict_2023_indicator_values_UnNorm(self) -> pd.Series:
#         """
#         用已训练的 self.mlp_model 预测 2023 指标变化量，并由 x_2022 + delta 得到 x_2023
#         返回：Series，index=指标，value=预测的2024指标值
#         """
#         if self.mlp_model is None:
#             raise RuntimeError("请先调用 ml_adjustment() 训练模型。")
#
#         data_matrix = self.data.values.astype(float)
#         # data_matrix = self.
#         x_2022 = data_matrix[:, -1]  # 2023指标向量（归一化后，理论在[0,1]）
#         delta_2023 = self.mlp_model.predict(x_2022.reshape(1, -1))[0]
#
#         # # ✅ 限制变化量幅度，避免一步跳太大（可显著减少越界概率）
#         delta_2023 = np.tanh(delta_2023)
#
#         x_2023_pred = x_2022 + delta_2023
#         # 若你希望约束为非负（可选）：x_2023_pred = np.clip(x_2023_pred, 0, None)
#         # ✅ 关键：归一化指标的可行域约束
#         # x_2023_pred = np.clip(x_2023_pred, 0.0, 1.0)
#         x_2023_pred = np.clip(x_2023_pred, 0.0, None)
#         return pd.Series(x_2023_pred, index=self.data.index, name="Predicted2023Value")
#
#         # ============================================================
#         # ✅ 新增 B：将2014-2023（含预测）重新计算熵权与MLP优化
#         # ============================================================
#
#     def recompute_weights_with_2023(self, group_name, alpha_opt) -> tuple[pd.Series, pd.Series]:
#         """
#         基于扩展数据(2014-2023)：先熵权，再用MLP（同样的差分训练逻辑）做权重优化
#         返回：
#           w_entropy_2014_2023, w_optimized_2014_2023
#         """
#         # 1) 预测2023指标值
#         x_2023 = self.predict_2023_indicator_values()
#
#         # 2) 组成扩展数据 2014-2023
#         year_2023 = "2023"
#         data_ext = self.data.copy()
#         # 如果列是int年份，保持一致
#         if len(self.years) > 0 and isinstance(self.years[-1], (int, np.integer)):
#             year_2023 = 2023
#         data_ext[year_2023] = x_2023.values
#         if group_name == "Group1":
#             # Group1 特定的归一化策略
#             for row in data_ext.index:
#                 if row == '人口密度 人/km2':
#                     data_ext.loc[row, :] = (data_ext.loc[row, :].max() - data_ext.loc[row, :]) / (
#                             data_ext.loc[row, :].max() - data_ext.loc[row, :].min())
#                 else:  # 其他行正向归一化
#                     data_ext.loc[row, :] = (data_ext.loc[row, :] - data_ext.loc[row, :].min()) / (
#                             data_ext.loc[row, :].max() - data_ext.loc[row, :].min())
#         elif group_name == "Group2":
#             # Group2 特定的归一化策略
#             for row in data_ext.index:
#                 if row == '工业废水排放量（万吨）' or row == '二氧化硫排放量（吨）':  # 第 5 行和第 6 行反向归一化
#                     data_ext.loc[row, :] = (data_ext.loc[row, :].max() - data_ext.loc[row, :]) / (
#                             data_ext.loc[row, :].max() - data_ext.loc[row, :].min())
#                 else:  # 其他行正向归一化
#                     data_ext.loc[row, :] = (data_ext.loc[row, :] - data_ext.loc[row, :].min()) / (
#                             data_ext.loc[row, :].max() - data_ext.loc[row, :].min())
#
#         # 3) 重新计算熵权（基于2014-2023）
#         w_entropy_ext = self.entropy_weight(data_ext)
#
#         # 4) 用扩展数据训练差分MLP，并优化权重（不影响你原流程，只是“追加结果”）
#         X = []
#         y = []
#         data_matrix = data_ext.values.astype(float)
#         for t in range(1, data_matrix.shape[1]):
#             prev_year = data_matrix[:, t - 1]
#             current_year = data_matrix[:, t]
#             X.append(prev_year)
#             y.append(current_year - prev_year)
#
#         X = np.array(X)
#         y = np.array(y)
#
#         model_ext = MLPRegressor(
#             hidden_layer_sizes=(12, 6),
#             activation="relu",
#             max_iter=2000,
#             random_state=42
#         )
#         model_ext.fit(X, y)
#
#         # 用2023前一年（即2022）作为输入，产生“末端趋势”来优化扩展熵权
#         last_data_ext = data_matrix[:, -2].reshape(1, -1)  # 仍用倒数第二列（2021）
#         delta_pred_ext = np.tanh(model_ext.predict(last_data_ext)[0])
#
#         w_opt_ext = w_entropy_ext.values * (1 + alpha_opt * delta_pred_ext)
#         # w_opt_ext = w_entropy_ext.values + alpha_opt * delta_pred_ext
#         w_opt_ext = np.clip(w_opt_ext, 0, None)
#         w_opt_ext = w_opt_ext / w_opt_ext.sum()
#
#         w_opt_ext = pd.Series(w_opt_ext, index=data_ext.index, name="OptimizedWeight_2014_2022")
#         w_entropy_ext.name = "EntropyWeight_2014_2022"
#
#         return w_entropy_ext, w_opt_ext
#
#     # ===============================
#     # **** 预测2023年权重（由2014-2022年数据计算-结束）****
#     # ===============================
#
# # ===============================
# # **** run_city（由2014-2022年数据计算-开始）****
# # ===============================
#
# # def run_city(city_name: str, group1_df: pd.DataFrame, group2_df: pd.DataFrame, group1_df_raw: pd.DataFrame, group2_df_raw: pd.DataFrame):
# #     """
# #     不改变原有流程：熵权 -> MLP优化 -> 预测2023年权重
# #     然后追加：
# #       - 预测2023指标值
# #       - 基于2014-2023重新算熵权与优化权重
# #     """
# #     results = {}
# #
# #     for group_name, df, dfraw in zip(["Group1", "Group2"], [group1_df, group2_df], [group1_df_raw, group2_df_raw]):
# #         model = CityWeightModel(df)
# #         model = CityWeightModel(dfraw)
# #
# #         # ===== 原流程 =====
# #         w_entropy = model.entropy_weight()  # 2014-2022 熵权
# #         model.initial_weights = w_entropy
# #         w_optimized = model.ml_adjustment(alpha=0.15)  # 2014-2022 MLP优化权重（并训练mlp_model）
# #         w_2023_weight = model.predict_2023_weights(alpha=0.15)  # 预测2023年权重
# #
# #         result_df = pd.DataFrame({
# #             "Indicator": df.index,
# #             "Entropy_Weight_2014_2022": w_entropy.values,
# #             "Optimized_Weight_2014_2022": w_optimized.values,
# #             "Predicted_2023_Weight": w_2023_weight.values
# #         })
# #
# #         # ===== 追加需求 A：预测2023指标值 =====
# #         pred_2023_values = model.predict_2023_indicator_values()
# #         result_df["Predicted_2023_IndicatorValue"] = pred_2023_values.values
# #
# #         # ===== 追加需求 B：基于2014-2023重新计算与优化权重 =====
# #         w_entropy_ext, w_opt_ext = model.recompute_weights_with_2023(group_name, alpha_opt=0.15)
# #         result_df["Entropy_Weight_2014_2023"] = w_entropy_ext.values
# #         result_df["Optimized_Weight_2014_2023"] = w_opt_ext.values
# #
# #         # ===== 追加需求 C：预测2023年未归一化的指标值 =====
# #         pred_2023_UnNormvalues = model.predict_2023_indicator_values_UnNorm()
# #         result_df["Predicted_2023_UnNormIndicatorValue"] = pred_2023_UnNormvalues.values
# #
# #         results[group_name] = result_df
# #
# #     return results
# def run_city(city_name: str, group1_df: pd.DataFrame, group2_df: pd.DataFrame):
#     """
#     不改变原有流程：熵权 -> MLP优化 -> 预测2023年权重
#     然后追加：
#       - 预测2023指标值
#       - 基于2014-2023重新算熵权与优化权重
#     """
#     results = {}
#
#     for group_name, df in zip(["Group1", "Group2"], [group1_df, group2_df]):
#         model = CityWeightModel(df)
#
#         # ===== 原流程 =====
#         w_entropy = model.entropy_weight()  # 2014-2022 熵权
#         model.initial_weights = w_entropy
#         w_optimized = model.ml_adjustment(alpha=0.15)  # 2014-2022 MLP优化权重（并训练mlp_model）
#         w_2023_weight = model.predict_2023_weights(alpha=0.15)  # 预测2023年权重
#
#         result_df = pd.DataFrame({
#             "Indicator": df.index,
#             "Entropy_Weight_2014_2022": w_entropy.values,
#             "Optimized_Weight_2014_2022": w_optimized.values,
#             "Predicted_2023_Weight": w_2023_weight.values
#         })
#
#         # ===== 追加需求 A：预测2023指标值 =====
#         pred_2023_values = model.predict_2023_indicator_values()
#         result_df["Predicted_2023_IndicatorValue"] = pred_2023_values.values
#
#         # ===== 追加需求 B：基于2014-2023重新计算与优化权重 =====
#         w_entropy_ext, w_opt_ext = model.recompute_weights_with_2023(group_name, alpha_opt=0.15)
#         result_df["Entropy_Weight_2014_2023"] = w_entropy_ext.values
#         result_df["Optimized_Weight_2014_2023"] = w_opt_ext.values
#
#         results[group_name] = result_df
#
#     return results
# # ===============================
# # **** run_city（由2014-2022年数据计算-结束）****
# # ===============================


# ==========================================
# 主程序（多城市，多sheet）
# ==========================================
if __name__ == "__main__":
    input_file = "城市数据熵权v4.xlsx"
    output_file = "All_Cities_Results_with_2024_values_and_reweights×0.14_0511.xlsx"
    # output_file = "All_Cities_Results_with_2023_values_and_reweights×0.15.xlsx"
    xls = pd.ExcelFile(input_file)
    sheet_names = xls.sheet_names

    # 原始数据读取
    # input_fileraw = "城市数据原始数据.xlsx"

    with pd.ExcelWriter(output_file) as writer:
        for city_name in sheet_names:
            print(f"正在处理城市：{city_name}")

            df = pd.read_excel(input_file, sheet_name=city_name, index_col=0)
            # dfraw = pd.read_excel(input_fileraw, sheet_name=city_name, index_col=0)

            # 若你是2014-2023共10年，请改为 0:10
            group1_df = df.iloc[0:10, 0:10]
            group2_df = df.iloc[11:21, 0:10]

            # # rawdata acquire
            # group1_df_raw = dfraw.iloc[0:10, 0:9]
            # group2_df_raw = dfraw.iloc[11:21, 0:9]

            results = run_city(city_name, group1_df, group2_df)
            # results = run_city(city_name, group1_df, group2_df, group1_df_raw, group2_df_raw)

            for group_name, result_df in results.items():
                sheet_out_name = f"{city_name}_{group_name}"
                result_df.to_excel(writer, sheet_name=sheet_out_name, index=False)

    print("全部城市计算完成，结果已输出。")


# # =========================
# # 0. EWM 和 MLP 权重对比分析图绘制
# # =========================
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import spearmanr
#
# # =========================
# # 1. Input data
# # =========================
# data = {
#     "Zhuhai": {
#         "ewm": [0.125421869,0.132604999,0.08361175,0.125477464,0.063980812,0.095092084,0.055741724,0.082971435,0.137770317,0.097327545,
#                 0.116079491,0.143061449,0.047504981,0.202279486,0.080165818,0.042506147,0.045306766,0.067605714,0.161422335,0.094067813],
#         "mlp": [0.126003168,0.133562671,0.084773395,0.127379699,0.06301628,0.092894658,0.055654193,0.081899604,0.136737265,0.098079067,
#                 0.115948337,0.143117868,0.047934025,0.204767702,0.078755083,0.042504481,0.045230096,0.067442354,0.160263269,0.094036786]
#     },
#     "Xiamen": {
#         "ewm": [0.135592991,0.087642002,0.085155207,0.153773957,0.049193749,0.075402157,0.165679894,0.088599413,0.073185994,0.085774636,
#                 0.134130724,0.187019432,0.05101027,0.08497522,0.053625004,0.049772007,0.071047581,0.059969841,0.171584564,0.136865355],
#         "mlp": [0.139846465,0.087769125,0.08590089,0.154904926,0.048064935,0.077097313,0.161273976,0.087919054,0.072473017,0.084750298,
#                 0.134381223,0.186289742,0.050974995,0.084539993,0.054838383,0.050623441,0.070750123,0.061209754,0.170213211,0.136179136]
#     },
#     "Fuzhou": {
#         "ewm": [0.112238155,0.058040613,0.118094582,0.121372636,0.103839333,0.104774605,0.089823462,0.146445807,0.055324508,0.090046299,
#                 0.103973294,0.13020525,0.074510169,0.117483323,0.120820059,0.04315935,0.062842371,0.085192006,0.133877629,0.127936549],
#         "mlp": [0.112067459,0.058178299,0.121442523,0.122653112,0.101076628,0.10477903,0.091505992,0.144597619,0.055131623,0.088567715,
#                 0.104070804,0.130898186,0.075526617,0.11864099,0.119481439,0.042290261,0.062402387,0.08554367,0.132252406,0.128893239]
#     },
#     "Shanghai": {
#         "ewm": [0.101748605,0.076560717,0.179024553,0.102287777,0.071666126,0.106777558,0.144837828,0.059353638,0.063898072,0.093845127,
#                 0.138532471,0.094980355,0.079646332,0.130697751,0.072371727,0.104761496,0.091258745,0.078561114,0.083756562,0.125433448],
#         "mlp": [0.103767385,0.077499764,0.179091659,0.102181083,0.070974929,0.110253403,0.142670249,0.058528175,0.062643587,0.092389764,
#                 0.138332183,0.095152259,0.080460619,0.131598728,0.071992703,0.104441676,0.091088013,0.0786262,0.082475312,0.125832307]
#     },
#     "Qingdao": {
#         "ewm": [0.115345699,0.085192665,0.208346557,0.113987227,0.070668898,0.079921577,0.081822462,0.108973115,0.064451246,0.071290554,
#                 0.100960618,0.124001585,0.13389118,0.142797844,0.094248637,0.052981502,0.067969604,0.079496233,0.07663239,0.127020406],
#         "mlp": [0.117226496,0.084975882,0.209525611,0.114864823,0.070720654,0.080975754,0.081542672,0.106259886,0.063395068,0.070513155,
#                 0.09928224,0.125002763,0.13809836,0.141020614,0.094998553,0.051987402,0.068886277,0.080005401,0.075674799,0.125043591]
#     },
#     "Qinhuangdao": {
#         "ewm": [0.066684247,0.095522383,0.118864032,0.070281317,0.146307622,0.083167921,0.107257133,0.187618657,0.05409562,0.070201068,
#                 0.098423967,0.096284867,0.08860394,0.234437242,0.066253462,0.081412524,0.062131648,0.053084083,0.114498322,0.104869943],
#         "mlp": [0.067519924,0.095037069,0.119073431,0.070770463,0.147787111,0.083837957,0.106739159,0.18561954,0.053038159,0.070577186,
#                 0.098135325,0.096950171,0.0895148,0.233447889,0.066253557,0.081121423,0.06255981,0.053153597,0.113098955,0.105764472]
#     },
#     "Dalian": {
#         "ewm": [0.030464819,0.197471555,0.124872716,0.033173668,0.04101376,0.261916619,0.064452221,0.159967255,0.045586207,0.041081181,
#                 0.053415671,0.155250894,0.067459693,0.073591364,0.126426621,0.050536088,0.067310321,0.074896705,0.175121016,0.155991626],
#         "mlp": [0.030531245,0.200347693,0.127763236,0.033369233,0.040361841,0.257238494,0.064596194,0.158943872,0.045186273,0.041661919,
#                 0.05289656,0.156813523,0.06707987,0.070080447,0.128214148,0.050641825,0.067315544,0.075287006,0.175974714,0.155696363]
#     }
# }
#
# cities = list(data.keys())
# n_indicators = 20
# # indicator_labels = [f"Ind{i}" for i in range(1, n_indicators + 1)]
# indicator_labels = [str(i) for i in range(1,21)]
#
# # =========================
# # 2. Compute metrics
# # =========================
# records = []
# all_ewm = []
# all_mlp = []
#
# for city in cities:
#     ewm = np.array(data[city]["ewm"], dtype=float)
#     mlp = np.array(data[city]["mlp"], dtype=float)
#
#     mean_abs_change = np.mean(np.abs(mlp - ewm))
#     rho, _ = spearmanr(ewm, mlp)
#     var_ewm = np.var(ewm)
#     var_mlp = np.var(mlp)
#     delta_var = var_mlp - var_ewm
#     delta_var_pct = (delta_var / var_ewm) * 100
#     max_abs_change = np.max(np.abs(mlp - ewm))
#
#     records.append([
#         city, mean_abs_change, rho, var_ewm, var_mlp, delta_var, delta_var_pct, max_abs_change
#     ])
#
#     all_ewm.extend(ewm)
#     all_mlp.extend(mlp)
#
# results_df = pd.DataFrame(
#     records,
#     columns=[
#         "City", "MeanAbsChange", "SpearmanRho",
#         "VarEWM", "VarMLP", "DeltaVar", "DeltaVarPct", "MaxAbsChange"
#     ]
# )
#
# print("\n=== City-level metrics ===")
# print(results_df.round(6).to_string(index=False))
#
# all_ewm = np.array(all_ewm)
# all_mlp = np.array(all_mlp)
# overall_rho, _ = spearmanr(all_ewm, all_mlp)
# overall_mean_abs_change = np.mean(np.abs(all_mlp - all_ewm))
# overall_var_ewm = np.var(all_ewm)
# overall_var_mlp = np.var(all_mlp)
# overall_delta_var = overall_var_mlp - overall_var_ewm
# overall_delta_var_pct = (overall_delta_var / overall_var_ewm) * 100
#
# print("\n=== Overall metrics ===")
# print(f"Overall MeanAbsChange = {overall_mean_abs_change:.6f}")
# print(f"Overall SpearmanRho   = {overall_rho:.6f}")
# print(f"Overall Var(EWM)      = {overall_var_ewm:.6f}")
# print(f"Overall Var(MLP)      = {overall_var_mlp:.6f}")
# print(f"Overall DeltaVar      = {overall_delta_var:.8f}")
# print(f"Overall DeltaVarPct   = {overall_delta_var_pct:.3f}%")
#
# # =========================
# # 3. Global style
# # =========================
# plt.rcParams["font.family"] = "DejaVu Serif"
# plt.rcParams["font.size"] = 10
# plt.rcParams["axes.linewidth"] = 0.8
#
# # # =========================
# # # Figure 1: Scatter consistency
# # # =========================
# # fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
# # axes = axes.flatten()
# #
# # for i, city in enumerate(cities):
# #     ax = axes[i]
# #     ewm = np.array(data[city]["ewm"])
# #     mlp = np.array(data[city]["mlp"])
# #     rho, _ = spearmanr(ewm, mlp)
# #
# #     ax.scatter(ewm, mlp, s=28, alpha=0.9)
# #     min_v = min(ewm.min(), mlp.min())
# #     max_v = max(ewm.max(), mlp.max())
# #     ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)
# #
# #     ax.set_title(city, fontsize=11)
# #     ax.text(0.05, 0.92, rf"$\rho$={rho:.3f}", transform=ax.transAxes, fontsize=9)
# #     ax.set_xlabel("EWM weight")
# #     ax.set_ylabel("MLP-optimized weight")
# #     ax.tick_params(length=3)
# #
# # # # Last panel: overall
# # # ax = axes[-1]
# # # ax.scatter(all_ewm, all_mlp, s=18, alpha=0.8)
# # # min_v = min(all_ewm.min(), all_mlp.min())
# # # max_v = max(all_ewm.max(), all_mlp.max())
# # # ax.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=1)
# # # ax.set_title("Overall", fontsize=11)
# # # ax.text(0.05, 0.92, rf"$\rho$={overall_rho:.3f}", transform=ax.transAxes, fontsize=9)
# # # ax.set_xlabel("EWM weight")
# # # ax.set_ylabel("MLP-optimized weight")
# # # ax.tick_params(length=3)
# # #
# # # fig.savefig("Figure1_scatter_consistency.png", dpi=600, bbox_inches="tight")
# #
# # # Last panel: overall
# # ax = axes[-1]
# #
# # # Scatter points
# # scatter_handle = ax.scatter(
# #     all_ewm,
# #     all_mlp,
# #     s=18,
# #     alpha=0.8,
# #     label="City-level indicator weights"
# # )
# #
# # # 1:1 reference line
# # min_v = min(all_ewm.min(), all_mlp.min())
# # max_v = max(all_ewm.max(), all_mlp.max())
# #
# # line_handle, = ax.plot(
# #     [min_v, max_v],
# #     [min_v, max_v],
# #     linestyle="--",
# #     linewidth=1,
# #     label="1:1 reference line"
# # )
# #
# # ax.set_title("Overall", fontsize=11)
# # ax.text(
# #     0.05,
# #     0.92,
# #     rf"$\rho$={overall_rho:.3f}",
# #     transform=ax.transAxes,
# #     fontsize=9
# # )
# #
# # ax.set_xlabel("EWM weight")
# # ax.set_ylabel("MLP-optimized weight")
# # ax.tick_params(length=3)
# #
# # # Add a unified legend for the whole figure
# # fig.legend(
# #     handles=[scatter_handle, line_handle],
# #     labels=["Indicator weights", "Equality line"],
# #     loc="lower center",
# #     ncol=2,
# #     frameon=False,
# #     fontsize=11,
# #     bbox_to_anchor=(0.5, 0.015)
# # )
# #
# # # Adjust layout to leave space for the legend
# # fig.tight_layout(rect=[0, 0.04, 1, 1])
# #
# # fig.savefig(
# #     "Figure1_scatter_consistency.png",
# #     dpi=600,
# #     bbox_inches="tight"
# # )
# #
# # # =========================
# # # Figure 2: Heatmap of delta weights
# # # =========================
# # delta_matrix = np.array([np.array(data[city]["mlp"]) - np.array(data[city]["ewm"]) for city in cities])
# #
# # fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
# # im = ax.imshow(delta_matrix, aspect="auto")
# # ax.set_xticks(np.arange(n_indicators))
# # ax.set_xticklabels(indicator_labels, rotation=45, ha="right")
# # ax.set_yticks(np.arange(len(cities)))
# # ax.set_yticklabels(cities)
# # ax.set_xlabel("Indicator")
# # ax.set_ylabel("City")
# # ax.set_title(r"Weight adjustment heatmap ($\Delta w = w^{MLP} - w^{EWM}$)")
# # cbar = fig.colorbar(im, ax=ax)
# # cbar.set_label(r"$\Delta w$")
# # fig.savefig("Figure2_heatmap_delta_weights.png", dpi=600, bbox_inches="tight")
# #
# # # =========================
# # # Figure 3: Boxplots of delta weights by city
# # # =========================
# # fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
# # delta_list = [np.array(data[city]["mlp"]) - np.array(data[city]["ewm"]) for city in cities]
# # ax.boxplot(delta_list, labels=cities, showfliers=True)
# # ax.axhline(0, linestyle="--", linewidth=1)
# # ax.set_ylabel(r"$\Delta w$")
# # ax.set_xlabel("City")
# # ax.set_title("Distribution of weight adjustments by city")
# # plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
# # fig.savefig("Figure3_boxplot_delta_weights.png", dpi=600, bbox_inches="tight")
# #
# # # =========================
# # # Figure 4: Summary metrics
# # # =========================
# # fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
# #
# # # (a) Mean absolute change
# # axes[0].bar(results_df["City"], results_df["MeanAbsChange"])
# # axes[0].set_title("(a) Mean absolute change")
# # axes[0].set_ylabel("Mean abs. change")
# # axes[0].tick_params(axis="x", rotation=25)
# #
# # # (b) Spearman rho
# # axes[1].bar(results_df["City"], results_df["SpearmanRho"])
# # axes[1].set_title("(b) Spearman correlation")
# # axes[1].set_ylabel(r"Spearman $\rho$")
# # axes[1].set_ylim(0.99, 1.001)
# # axes[1].tick_params(axis="x", rotation=25)
# #
# # # (c) Delta variance percentage
# # axes[2].bar(results_df["City"], results_df["DeltaVarPct"])
# # axes[2].axhline(0, linestyle="--", linewidth=1)
# # axes[2].set_title("(c) Variance change (%)")
# # axes[2].set_ylabel(r"$\Delta Var$ (%)")
# # axes[2].tick_params(axis="x", rotation=25)
# #
# # fig.savefig("Figure4_summary_metrics.png", dpi=600, bbox_inches="tight")
# #
# # plt.show()
#
# # =========================
# # Figure 1: Scatter consistency
# # =========================
#
# # 不使用 constrained_layout，避免与 tight_layout 冲突
# fig, axes = plt.subplots(2, 4, figsize=(14, 7))
# axes = axes.flatten()
#
# for i, city in enumerate(cities):
#     ax = axes[i]
#
#     ewm = np.array(data[city]["ewm"])
#     mlp = np.array(data[city]["mlp"])
#     rho, _ = spearmanr(ewm, mlp)
#
#     # Scatter points
#     ax.scatter(
#         ewm,
#         mlp,
#         s=28,
#         alpha=0.9
#     )
#
#     # 1:1 equality line
#     min_v = min(ewm.min(), mlp.min())
#     max_v = max(ewm.max(), mlp.max())
#
#     # 增加少量边距，避免点和线贴边
#     pad = (max_v - min_v) * 0.05 if max_v > min_v else 0.001
#     plot_min = min_v - pad
#     plot_max = max_v + pad
#
#     ax.plot(
#         [plot_min, plot_max],
#         [plot_min, plot_max],
#         linestyle="--",
#         linewidth=1
#     )
#
#     ax.set_xlim(plot_min, plot_max)
#     ax.set_ylim(plot_min, plot_max)
#
#     ax.set_title(city, fontsize=11)
#     ax.text(
#         0.05,
#         0.92,
#         rf"$\rho$={rho:.3f}",
#         transform=ax.transAxes,
#         fontsize=9
#     )
#
#     ax.set_xlabel("EWM weight")
#     ax.set_ylabel("MLP-optimized weight")
#     ax.tick_params(length=3)
#
#
# # =========================
# # Last panel: Overall
# # =========================
# ax = axes[-1]
#
# # Scatter points
# scatter_handle = ax.scatter(
#     all_ewm,
#     all_mlp,
#     s=18,
#     alpha=0.8,
#     label="Indicator weights"
# )
#
# # 1:1 equality line
# min_v = min(all_ewm.min(), all_mlp.min())
# max_v = max(all_ewm.max(), all_mlp.max())
#
# pad = (max_v - min_v) * 0.05 if max_v > min_v else 0.001
# plot_min = min_v - pad
# plot_max = max_v + pad
#
# line_handle, = ax.plot(
#     [plot_min, plot_max],
#     [plot_min, plot_max],
#     linestyle="--",
#     linewidth=1,
#     label="Equality line"
# )
#
# ax.set_xlim(plot_min, plot_max)
# ax.set_ylim(plot_min, plot_max)
#
# ax.set_title("Overall", fontsize=11)
# ax.text(
#     0.05,
#     0.92,
#     rf"$\rho$={overall_rho:.3f}",
#     transform=ax.transAxes,
#     fontsize=9
# )
#
# ax.set_xlabel("EWM weight")
# ax.set_ylabel("MLP-optimized weight")
# ax.tick_params(length=3)
#
#
# # =========================
# # Unified legend
# # =========================
# fig.legend(
#     handles=[scatter_handle, line_handle],
#     labels=["Indicator weights", "Equality line"],
#     loc="lower center",
#     ncol=2,
#     frameon=False,
#     fontsize=11,
#     bbox_to_anchor=(0.5, 0.015)
# )
#
# # 给底部统一图例预留空间
# fig.tight_layout(rect=[0, 0.08, 1, 1])
#
# fig.savefig(
#     "Figure1_scatter_consistency.png",
#     dpi=600,
#     bbox_inches="tight"
# )
#
# # =========================
# # Figure 2: Heatmap of delta weights
# # =========================
# delta_matrix = np.array([np.array(data[city]["mlp"]) - np.array(data[city]["ewm"]) for city in cities])
#
# fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
# im = ax.imshow(delta_matrix, aspect="auto")
# ax.set_xticks(np.arange(n_indicators))
# ax.set_xticklabels(indicator_labels, rotation=45, ha="right")
# ax.set_yticks(np.arange(len(cities)))
# ax.set_yticklabels(cities)
# ax.set_xlabel("Indicator")
# ax.set_ylabel("City")
# ax.set_title(r"Weight adjustment heatmap ($\Delta w = w^{MLP} - w^{EWM}$)")
# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(r"$\Delta w$")
# fig.savefig("Figure2_heatmap_delta_weights.png", dpi=600, bbox_inches="tight")
#
# # =========================
# # Figure 3: Boxplots of delta weights by city
# # =========================
# fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
# delta_list = [np.array(data[city]["mlp"]) - np.array(data[city]["ewm"]) for city in cities]
# ax.boxplot(delta_list, labels=cities, showfliers=True)
# ax.axhline(0, linestyle="--", linewidth=1)
# ax.set_ylabel(r"$\Delta w$")
# ax.set_xlabel("City")
# ax.set_title("Distribution of weight adjustments by city")
# plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
# fig.savefig("Figure3_boxplot_delta_weights.png", dpi=600, bbox_inches="tight")
#
# # =========================
# # Figure 4: Summary metrics
# # =========================
# fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
#
# # (a) Mean absolute change
# axes[0].bar(results_df["City"], results_df["MeanAbsChange"])
# axes[0].set_title("(a) Mean absolute change")
# axes[0].set_ylabel("Mean abs. change")
# axes[0].tick_params(axis="x", rotation=25)
#
# # (b) Spearman rho
# axes[1].bar(results_df["City"], results_df["SpearmanRho"])
# axes[1].set_title("(b) Spearman correlation")
# axes[1].set_ylabel(r"Spearman $\rho$")
# axes[1].set_ylim(0.99, 1.001)
# axes[1].tick_params(axis="x", rotation=25)
#
# # (c) Delta variance percentage
# axes[2].bar(results_df["City"], results_df["DeltaVarPct"])
# axes[2].axhline(0, linestyle="--", linewidth=1)
# axes[2].set_title("(c) Variance change (%)")
# axes[2].set_ylabel(r"$\Delta Var$ (%)")
# axes[2].tick_params(axis="x", rotation=25)
#
# fig.savefig("Figure4_summary_metrics.png", dpi=600, bbox_inches="tight")
# plt.show()