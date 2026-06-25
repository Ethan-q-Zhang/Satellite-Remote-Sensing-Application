import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 1. 创建完整数据集
data = {
    'year': [2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'X1': [0.0001, 0.1528, 0.3328, 0.5424, 0.4866, 0.6279, 1.0001],
    'X2': [0.0001, 0.0924, 0.167, 0.2569, 0.3474, 0.8702, 1.0001],
    'X3': [0.0001, 0.391, 0.6128, 0.9013, 0.883, 0.5909, 1.0001],
    'X4': [0.0001, 0.2041, 0.2773, 0.4167, 0.828, 0.9021, 1.0001],
    'X5': [0.0001, 0.3334, 0.6668, 1.0001, 0.5001, 0.6668, 0.8334],
    'X6': [0.0001, 0.2167, 0.4621, 0.615, 0.7539, 0.6118, 1.0001],
    'X7': [0.6001, 1.0001, 0.8001, 0.5001, 0.4001, 0.0001, 0.0001],
    'X8': [0.0001, 0.0825, 0.2001, 0.4119, 0.6472, 0.8236, 1.0001],
    'X9': [0.0001, 0.1528, 0.3326, 0.5423, 0.4865, 0.6278, 1.0001],
    'X10': [1.0001, 0.9078, 0.8332, 0.7433, 0.6528, 0.13, 0.0001],
    'X11': [0.4305, 0.9242, 1.0001, 0.3292, 0.3672, 0.3292, 0.0001],
    'X12': [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0001, 1.0001],
    'X13': [1.0001, 0.6801, 0.7801, 0.4801, 0.7001, 0.4601, 0.0001],
    'X14': [0.3187, 0.9092, 1.0001, 0.2759, 0.3541, 0.3309, 0.0001],
    'X15': [1.0001, 0.909, 0.4564, 0.9233, 0.0001, 0.0709, 0.5946],
    'X16': [0.1668, 0.304, 0.755, 1.0001, 0.0001, 0.755, 0.6177],
    'X17': [0.1983, 0.0001, 0.1105, 0.1442, 0.374, 0.5496, 1.0001],
    'X18': [0.0001, 0.9168, 0.6668, 0.4376, 0.5626, 1.0001, 0.7918],
    'X19': [0.0001, 0.9715, 1.0001, 0.9287, 0.8944, 0.9287, 0.9565],
    'X20': [0.0001, 0.7346, 0.983, 1.0001, 0.9499, 0.9668, 0.8043],
    'X21': [0.0001, 0.7835, 0.8734, 0.8839, 0.8922, 0.9972, 1.0001],
    'CCD': [0.3288, 0.5406, 0.627, 0.6347, 0.4615, 0.7702, 0.9009]
}

# 创建DataFrame
df = pd.DataFrame(data)
df.set_index('year', inplace=True)

# 2. 定义UDL和EEQ指标
udl_features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']

eeq_features = ['X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21']

# 3. 分离特征和目标变量
X = df.drop('CCD', axis=1)
y = df['CCD']

# 4. 优化后的快速随机森林分析函数
def fast_random_forest_analysis(X_features, y_target, feature_names, group_name, n_repeats=50):
    """
    快速随机森林分析，针对小样本数据优化
    
    参数:
    X_features: 特征DataFrame
    y_target: 目标变量Series
    feature_names: 特征名称列表
    group_name: 组名
    n_repeats: 重复实验次数（减少到50次）
    """
    X_subset = X[X_features].copy()
    
    n_samples = len(y_target)
    n_features = len(feature_names)
    
    # 存储结果
    importance_matrix = []
    
    np.random.seed(42)
    
    for repeat in range(n_repeats):
        # 使用留一法：每次留出一个样本作为测试集
        test_idx = [repeat % n_samples]  # 循环使用每个样本作为测试
        train_idx = [i for i in range(n_samples) if i != test_idx[0]]
        
        X_train = X_subset.iloc[train_idx].values
        X_test = X_subset.iloc[test_idx].values
        y_train = y_target.iloc[train_idx].values
        y_test = y_target.iloc[test_idx].values
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 简化的随机森林模型
        rf = RandomForestRegressor(
            n_estimators=30,      # 减少树的数量
            max_depth=2,          # 限制树深度
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=repeat,
            n_jobs=1             # 不使用并行，减少资源占用
        )
        
        rf.fit(X_train_scaled, y_train)
        
        # 获取特征重要性
        if hasattr(rf, 'feature_importances_'):
            importance_matrix.append(rf.feature_importances_)
    
    # 计算平均特征重要性
    if importance_matrix:
        importance_matrix = np.array(importance_matrix)
        avg_importance = np.mean(importance_matrix, axis=0)
        std_importance = np.std(importance_matrix, axis=0)
    else:
        avg_importance = np.ones(n_features) / n_features
        std_importance = np.zeros(n_features)
    
    return {
        'features': feature_names,
        'avg_importance': avg_importance,
        'std_importance': std_importance,
        'group': group_name
    }

# 5. 执行三组分析
print("=" * 60)
print("Random Forest Analysis - Core Driving Factors of CCD")
print("=" * 60)

# 5.1 全指标分析
print("\n1. All indicators analysis (21 indicators)...")
all_features = udl_features + eeq_features
all_results = fast_random_forest_analysis(all_features, y, all_features, "All indicators", n_repeats=30)

# 5.2 UDL指标分析
print("2. Urban Development Level (UDL) analysis (10 indicators)...")
udl_results = fast_random_forest_analysis(udl_features, y, udl_features, "UDL", n_repeats=30)

# 5.3 EEQ指标分析
print("3. Ecological Economic Quality (EEQ) analysis (11 indicators)...")
eeq_results = fast_random_forest_analysis(eeq_features, y, eeq_features, "EEQ", n_repeats=30)

print("\n✓ Analysis completed!")

# 6. 创建重要性DataFrame
def create_importance_df(results):
    df = pd.DataFrame({
        'Feature': results['features'],
        'Mean of importance': results['avg_importance'],
        'STD of importance': results['std_importance']
    })
    
    df = df.sort_values('Mean of importance', ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    df['Cumulative_Importance'] = df['Mean of importance'].cumsum()
    
    return df

all_df = create_importance_df(all_results)
udl_df = create_importance_df(udl_results)
eeq_df = create_importance_df(eeq_results)

# 7. 基本可视化
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 7.1 全指标重要性排名
ax1 = axes[0, 0]
top_all = all_df.head(10)
bars1 = ax1.barh(range(len(top_all)), top_all['Mean of importance'][::-1])
ax1.set_yticks(range(len(top_all)))
ax1.set_yticklabels(top_all['Feature'][::-1], fontsize=8)
ax1.set_xlabel('Feature importance')
ax1.set_title('Top-10 Feature Importance (All Indicators)')
ax1.invert_yaxis()

# 7.2 UDL指标重要性
ax2 = axes[0, 1]
bars2 = ax2.barh(range(len(udl_df)), udl_df['Mean of importance'][::-1])
ax2.set_yticks(range(len(udl_df)))
ax2.set_yticklabels(udl_df['Feature'][::-1], fontsize=8)
ax2.set_xlabel('Feature importance')
ax2.set_title('UDL Indicators Importance')
ax2.invert_yaxis()

# 7.3 EEQ指标重要性
ax3 = axes[0, 2]
bars3 = ax3.barh(range(len(eeq_df)), eeq_df['Mean of importance'][::-1])
ax3.set_yticks(range(len(eeq_df)))
ax3.set_yticklabels(eeq_df['Feature'][::-1], fontsize=8)
ax3.set_xlabel('Feature importance')
ax3.set_title('EEQ Indicators Importance')
ax3.invert_yaxis()

# 7.4 累积重要性对比
ax4 = axes[1, 0]
cumulative_all = all_df['Cumulative_Importance'].values
cumulative_udl = udl_df['Cumulative_Importance'].values
cumulative_eeq = eeq_df['Cumulative_Importance'].values

x_all = range(1, len(cumulative_all) + 1)
x_udl = range(1, len(cumulative_udl) + 1)
x_eeq = range(1, len(cumulative_eeq) + 1)

ax4.plot(x_all, cumulative_all, 'o-', label='All indicators', linewidth=2, markersize=4)
ax4.plot(x_udl, cumulative_udl, 's-', label='UDL', linewidth=2, markersize=4)
ax4.plot(x_eeq, cumulative_eeq, '^-', label='EEQ', linewidth=2, markersize=4)
ax4.axhline(y=0.8, color='r', linestyle='--', alpha=0.7)
ax4.set_xlabel('Number of features')
ax4.set_ylabel('Cumulative_Importance')
ax4.set_title('Cumulative Importance Comparison')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# 7.5 核心驱动因素总结
ax5 = axes[1, 1]
# 找出每组前3个最重要特征
top_features_summary = {
    'All indicators': all_df.head(3)['Feature'].tolist(),
    'UDL': udl_df.head(3)['Feature'].tolist(),
    'EEQ': eeq_df.head(3)['Feature'].tolist()
}

# 创建一个简单的文本总结
summary_text = "Core Driving Factors Summary:\n\n"
for group, features in top_features_summary.items():
    summary_text += f"{group}:\n"
    for i, feat in enumerate(features, 1):
        summary_text += f"  {i}. {feat}\n"
    summary_text += "\n"

ax5.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
ax5.axis('off')
ax5.set_title('Core Driving Factors Summary')

# 7.6 CCD时间序列
ax6 = axes[1, 2]
years = df.index
ccd_values = y.values
ax6.plot(years, ccd_values, 'o-', linewidth=2, markersize=8, color='red')
ax6.fill_between(years, ccd_values, alpha=0.2, color='red')
ax6.set_xlabel('year')
ax6.set_ylabel('CCD')
ax6.set_title('Time series changes of CCD')
ax6.grid(True, alpha=0.3)
ax6.set_xticks(years)

plt.suptitle('CCD Core Driving Factors Analysis (2015-2021)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('CCD Core Driving Factors Analysis (2015-2021).png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 结果输出
print("\n" + "="*60)
print("Analysis Results Summary")
print("="*60)

# 8.1 全指标Top-5
print("\n1. Top-5 Core Driving Factors (All Indicators):")
for i, row in all_df.head(5).iterrows():
    print(f"   {row['Rank']}. {row['Feature']} (Importance: {row['Mean of importance']:.4f})")

# 8.2 UDL Top-3
print("\n2. Top-3 Core Driving Factors (UDL):")
for i, row in udl_df.head(3).iterrows():
    print(f"   {row['Rank']}. {row['Feature']} (Importance: {row['Mean of importance']:.4f})")

# 8.3 EEQ Top-3
print("\n3. Top-3 Core Driving Factors (EEQ):")
for i, row in eeq_df.head(3).iterrows():
    print(f"   {row['Rank']}. {row['Feature']} (Importance: {row['Mean of importance']:.4f})")

# 8.4 解释80%变异所需特征数
print("\n4. Minimum Features to Explain 80% Variance:")
for df_group, name in [(all_df, "All indicators"), (udl_df, "UDL"), (eeq_df, "EEQ")]:
    cumulative = df_group['Cumulative_Importance'].values
    n_features_80 = np.where(cumulative >= 0.8)[0]
    if len(n_features_80) > 0:
        n_features_80 = n_features_80[0] + 1
    else:
        n_features_80 = len(cumulative)
    
    percentage = n_features_80 / len(df_group) * 100
    print(f"   {name}: {n_features_80}features ({percentage:.1f}% of total)")

# 9. 保存结果
try:
    all_df.to_csv('All_Indicators_Feature_Importance.csv', index=False, encoding='utf-8-sig')
    udl_df.to_csv('UDL_Feature_Importance.csv', index=False, encoding='utf-8-sig')
    eeq_df.to_csv('EEQ_Feature_Importance.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*60)
    print("Results Saved:")
    print("  1. All_Indicators_Feature_Importance.csv")
    print("  2. UDL_Feature_Importance.csv")
    print("  3. EEQ_Feature_Importance.csv")
    print("  4. CCD_Core_Driving_Factors_Analysis.png")
    print("="*60)
except Exception as e:
    print(f"\nError saving files: {str(e)}")

print("\n✓ Analysis completed successfully!")