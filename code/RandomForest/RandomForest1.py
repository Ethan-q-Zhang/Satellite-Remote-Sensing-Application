import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")

# 1. Create dataset from the table
# Zhuhai
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'X1': [0.0001, 0.1086, 0.2908, 0.4793, 0.76, 0.7848, 1.0001],
    'X2': [0.0001, 0.0496, 0.1578, 0.3088, 0.468, 0.9796, 1.0001],
    'X3': [1.0001, 0.9506, 0.8424, 0.6914, 0.5322, 0.0206, 0.0001],
    'X4': [0.0001, 0.1086, 0.2908, 0.4793, 0.76, 0.7848, 1.0001],
    'X5': [0.0001, 0.1751, 0.2431, 0.1972, 0.1384, 0.3172, 1.0001],
    'X6': [0.0001, 0.0915, 0.3858, 0.5982, 0.7207, 1.0001, 0.9254],
    'X7': [1.0001, 0.611, 0.637, 0.6392, 0.6361, 0.0001, 0.2675],
    'X8': [0.0001, 0.1112, 0.5557, 0.6668, 0.6668, 0.6668, 1.0001],
    'X9': [1.0001, 0.6853, 0.6297, 0.352, 0.3705, 0.1853, 0.0001],
    'X10': [0.0001, 0.152, 0.2469, 0.3988, 0.576, 0.6014, 1.0001],
    'X11': [1.0001, 0.9882, 0.8215, 0.6072, 0.4763, 0.3572, 0.0001],
    'X12': [1.0001, 0.8668, 0.7334, 0.5334, 0.5334, 0.4668, 0.0001],
    'X13': [0.1715, 0.0001, 0.2001, 0.6953, 0.7811, 0.943, 1.0001],
    'X14': [1.0001, 0.1516, 0.1978, 0.3186, 0.0892, 0.0001, 0.0487],
    'X15': [0.0001, 0.0747, 0.112, 0.1494, 0.6456, 0.9479, 1.0001],
    'X16': [0.0001, 0.2501, 0.2751, 0.2918, 0.3834, 0.6168, 1.0001],
    'X17': [0.0001, 0.7171, 0.7157, 0.7549, 0.8332, 1.0001, 0.9867],
    'X18': [0.0001, 0.9214, 0.9099, 0.8946, 0.9489, 0.9872, 1.0001],
    'X19': [0.4001, 0.9295, 0.2825, 0.2825, 0.0001, 0.8001, 1.0001],
    'X20': [1.0001, 0.9188, 0.771, 0.5269, 0.4688, 0.3181, 0.0001],
    'X21': [0.9612, 1.0001, 0.8395, 0.511, 0.3377, 0.2724, 0.0001],
    'X22': [0.2001, 1.0001, 0.6001, 0.0001, 0.6001, 0.2001, 0.0001],
    'X23': [0.2858, 0.5715, 0.5001, 0.2858, 0.0001, 0.3572, 1.0001],
    'CCD': [0.4043, 0.4474, 0.5787, 0.6464, 0.7316, 0.7827, 0.7859]
}

weights = {
    'X1': 0.1127, 'X2': 0.1619, 'X3': 0.0503, 'X4': 0.0986, 'X5': 0.1504,
    'X6': 0.1065, 'X7': 0.0497, 'X8': 0.0939, 'X9': 0.0767, 'X10': 0.0993,
    'X11': 0.0110, 'X12': 0.0099, 'X13': 0.0439, 'X14': 0.0609, 'X15': 0.0630,
    'X16': 0.0410, 'X17': 0.0207, 'X18': 0.0204, 'X19': 0.0348, 'X20': 0.0082,
    'X21': 0.0196, 'X22': 0.3333, 'X23': 0.3333
}

# Shanghai
# data = {
#     'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021],
#     'X1': [0.0001, 0.169, 0.3046, 0.4178, 0.7204, 0.7506, 1.0001],
#     'X2': [0.0001, 0.0598, 0.0414, 0.1149, 0.1736, 0.9857, 1.0001],
#     'X3': [1.0001, 0.9404, 0.9588, 0.8853, 0.8266, 0.0145, 0.0001],
#     'X4': [0.0001, 0.169, 0.3046, 0.4178, 0.7204, 0.7505, 1.0001],
#     'X5': [0.0001, 0.4331, 0.5515, 0.7739, 0.7956, 0.6532, 1.0001],
#     'X6': [0.0001, 0.1261, 0.2794, 0.3972, 0.5186, 0.7765, 1.0001],
#     'X7': [0.0001, 0.1074, 0.216, 0.3365, 0.7602, 0.7199, 1.0001],
#     'X8': [0.0001, 0.2382, 0.3334, 0.5715, 0.8096, 0.8096, 1.0001],
#     'X9': [0.4723, 0.4723, 0.3612, 0.6945, 1.0001, 0.389, 0.0001],
#     'X10': [0.0001, 0.0757, 0.2438, 0.4203, 0.5463, 0.7564, 1.0001],
#     'X11': [0.9137, 1.0001, 0.7038, 0.3705, 0.3087, 0.1606, 0.0001],
#     'X12': [1.0001, 1.0001, 1.0001, 0.0001, 0.0001, 0.0001, 0.0001],
#     'X13': [0.6668, 1.0001, 0.9068, 1.0001, 0.5601, 0.8134, 0.0001],
#     'X14': [0.0001, 0.0566, 0.1131, 0.1696, 0.2261, 0.2826, 1.0001],
#     'X15': [0.0001, 0.1819, 0.4546, 0.5456, 0.7274, 0.8183, 1.0001],
#     'X16': [0.0001, 0.4144, 0.0001, 0.6792, 0.7624, 0.6792, 1.0001],
#     'X17': [0.0001, 0.5788, 0.8597, 1.0001, 0.7192, 0.8821, 0.8316],
#     'X18': [0.0001, 0.6398, 0.9493, 0.9656, 0.9874, 1.0001, 0.9977],
#     'X19': [0.0001, 0.2228, 0.2181, 0.493, 0.6636, 0.7821, 1.0001],
#     'X20': [0.8865, 1.0001, 0.6892, 0.4395, 0.4826, 0.2152, 0.0001],
#     'X21': [1.0001, 0.5268, 0.4991, 0.139, 0.3299, 0.0001, 0.6446],
#     'X22': [0.5001, 0.0001, 0.0001, 1.0001, 1.0001, 0.5001, 1.0001],
#     'X23': [0.8890, 0.0001, 0.2223, 1.0001, 0.3334, 0.5557, 0.8890],
#     'CCD': [0.2783, 0.4196, 0.5013, 0.6115, 0.7582, 0.747, 0.7857]
# }
#
# weights = {
#     'X1': 0.0937, 'X2': 0.2275, 'X3': 0.0605, 'X4': 0.0993, 'X5': 0.0725,
#     'X6': 0.1117, 'X7': 0.1159, 'X8': 0.0842, 'X9': 0.0175, 'X10': 0.1172,
#     'X11': 0.0208, 'X12': 0.0673, 'X13': 0.0031, 'X14': 0.0529, 'X15': 0.0298,
#     'X16': 0.0498, 'X17': 0.0109, 'X18': 0.0195, 'X19': 0.0355, 'X20': 0.0098,
#     'X21': 0.0339, 'X22': 0.3333, 'X23': 0.3333
# }

#Qingdao
# data = {
#     'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021],
#     'X1': [0.0001, 0.1528, 0.3328, 0.5424, 0.4866, 0.6279, 1.0001],
#     'X2': [0.0001, 0.0924, 0.167, 0.2569, 0.3474, 0.8702, 1.0001],
#     'X3': [1.0001, 0.9078, 0.8332, 0.7433, 0.6528, 0.13, 0.0001],
#     'X4': [0.0001, 0.1528, 0.3326, 0.5423, 0.4865, 0.6278, 1.0001],
#     'X5': [0.0001, 0.391, 0.6128, 0.9013, 0.883, 0.5909, 1.0001],
#     'X6': [0.0001, 0.2041, 0.2773, 0.4167, 0.828, 0.9021, 1.0001],
#     'X7': [0.0001, 0.2167, 0.4621, 0.615, 0.7539, 0.6118, 1.0001],
#     'X8': [0.0001, 0.3334, 0.6668, 1.0001, 0.5001, 0.6668, 0.8334],
#     'X9': [0.6001, 1.0001, 0.8001, 0.5001, 0.4001, 0.0001, 0.0001],
#     'X10': [0.0001, 0.0825, 0.2001, 0.4119, 0.6472, 0.8236, 1.0001],
#     'X11': [0.4305, 0.9242, 1.0001, 0.3292, 0.3672, 0.3292, 0.0001],
#     'X12': [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1.0001, 1.0001],
#     'X13': [1.0001, 0.6801, 0.7801, 0.4801, 0.7001, 0.4601, 0.0001],
#     'X14': [0.1983, 0.0001, 0.1105, 0.1442, 0.374, 0.5496, 1.0001],
#     'X15': [0.0001, 0.9168, 0.6668, 0.4376, 0.5626, 1.0001, 0.7918],
#     'X16': [0.0001, 0.9715, 1.0001, 0.9287, 0.8944, 0.9287, 0.9565],
#     'X17': [0.0001, 0.7346, 0.983, 1.0001, 0.9499, 0.9668, 0.8043],
#     'X18': [0.0001, 0.7835, 0.8734, 0.8839, 0.8922, 0.9972, 1.0001],
#     'X19': [0.1668, 0.304, 0.755, 1.0001, 0.0001, 0.755, 0.6177],
#     'X20': [0.3187, 0.9092, 1.0001, 0.2759, 0.3541, 0.3309, 0.0001],
#     'X21': [1.0001, 0.909, 0.4564, 0.9233, 0.0001, 0.0709, 0.5946],
#     'X22': [0.7501, 0.8751, 1.0001, 0.7501, 0.0001, 0.5001, 0.7501],
#     'X23': [0.5557, 0.1112, 0.6668, 1.0001, 0.0001, 0.5557, 1.0001],
#     'CCD': [0.3258, 0.5078, 0.6016, 0.7048, 0.754, 0.7589, 0.9137]
# }
# weights = {
#     'X1': 0.1102, 'X2': 0.163, 'X3': 0.0473, 'X4': 0.1105, 'X5': 0.0853,
#     'X6': 0.1081, 'X7': 0.09, 'X8': 0.0704, 'X9': 0.1018, 'X10': 0.1134,
#     'X11': 0.0095, 'X12': 0.1230, 'X13': 0.0012, 'X14': 0.0452, 'X15': 0.0139,
#     'X16': 0.0200, 'X17': 0.0147, 'X18': 0.0159, 'X19': 0.0386, 'X20': 0.0121,
#     'X21': 0.0393, 'X22': 0.3333, 'X23': 0.3333
# }

# Create DataFrame
df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

# 2. Define UDL and EEQ indicators
udl_features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
eeq_features = ['X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23']

# 3. Separate features and target variable
X = df.drop('CCD', axis=1)
y = df['CCD']

# 4. Weights from the table
#Zhuhai
# weights = {
#     'X1': 0.1127, 'X2': 0.1619, 'X3': 0.1504, 'X4': 0.1065, 'X5': 0.0939,
#     'X6': 0.0497, 'X7': 0.0767, 'X8': 0.0993, 'X9': 0.0986, 'X10': 0.0503,
#     'X11': 0.0331, 'X12': 0.0297, 'X13': 0.1316, 'X14': 0.0246, 'X15': 0.0589,
#     'X16': 0.1043, 'X17': 0.1827, 'X18': 0.1889, 'X19': 0.1231, 'X20': 0.062,
#     'X21': 0.0611
# }

# 5. Fast Random Forest analysis function
def fast_random_forest_analysis(X_features, y_target, feature_names, group_name, n_repeats=30):
    """Fast Random Forest analysis optimized for small sample data"""
    X_subset = X[X_features].copy()
    
    n_samples = len(y_target)
    n_features = len(feature_names)
    
    # Store results
    importance_matrix = []
    
    np.random.seed(42)
    
    for repeat in range(n_repeats):
        # Leave-one-out: use each sample as test set in turn
        test_idx = [repeat % n_samples]
        train_idx = [i for i in range(n_samples) if i != test_idx[0]]
        
        X_train = X_subset.iloc[train_idx].values
        X_test = X_subset.iloc[test_idx].values
        y_train = y_target.iloc[train_idx].values
        y_test = y_target.iloc[test_idx].values
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Simplified Random Forest model
        rf = RandomForestRegressor(
            n_estimators=30,
            max_depth=2,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=repeat,
            n_jobs=1
        )
        
        rf.fit(X_train_scaled, y_train)
        
        # Get feature importance
        if hasattr(rf, 'feature_importances_'):
            importance_matrix.append(rf.feature_importances_)
    
    # Calculate average feature importance
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

# 6. Perform three-group analysis
print("=" * 60)
print("Random Forest Analysis - Core Driving Factors of CCD")
print("=" * 60)

# 6.1 All indicators analysis
print("\n1. All indicators analysis (23 indicators)...")
all_features = udl_features + eeq_features
all_results = fast_random_forest_analysis(all_features, y, all_features, "All Indicators", n_repeats=30)

# 6.2 UDL indicators analysis
print("2. Urban Development Level (UDL) analysis (10 indicators)...")
udl_results = fast_random_forest_analysis(udl_features, y, udl_features, "Urban Development Level (UDL)", n_repeats=30)

# 6.3 EEQ indicators analysis
print("3. Ecological Economic Quality (EEQ) analysis (13 indicators)...")
eeq_results = fast_random_forest_analysis(eeq_features, y, eeq_features, "Ecological Economic Quality (EEQ)", n_repeats=30)

print("\n✓ Analysis completed!")

# 7. Create importance DataFrame
def create_importance_df(results):
    df = pd.DataFrame({
        'Feature': results['features'],
        'Mean_Importance': results['avg_importance'],
        'Std_Importance': results['std_importance']
    })
    
    df = df.sort_values('Mean_Importance', ascending=False)
    df['Rank'] = range(1, len(df) + 1)
    df['Cumulative_Importance'] = df['Mean_Importance'].cumsum()
    
    return df

all_df = create_importance_df(all_results)
udl_df = create_importance_df(udl_results)
eeq_df = create_importance_df(eeq_results)

# 8. Basic visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 8.1 All indicators importance ranking
# ax1 = axes[0, 0]
# top_all = all_df.head(10)
# colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(top_all)))
# bars1 = ax1.barh(range(len(top_all)), top_all['Mean_Importance'][::-1])
# ax1.set_yticks(range(len(top_all)))
# ax1.set_yticklabels(top_all['Feature'][::-1], fontsize=8)
# ax1.set_xlabel('Feature Importance')
# ax1.set_title('Top-10 Feature Importance (All Indicators)')
# ax1.invert_yaxis()

ax1 = axes[0, 0]
top_all = all_df.head(10)
values1 = top_all['Mean_Importance'][::-1]

# 使用正确的颜色映射调用方式
cmap = plt.get_cmap('jet')
norm1 = plt.Normalize(values1.min(), values1.max())
colors1 = cmap(norm1(values1))
bars1 = ax1.barh(range(len(top_all)), values1, color=colors1)

ax1.set_yticks(range(len(top_all)))
ax1.set_yticklabels(top_all['Feature'][::-1], fontsize=8)
ax1.set_xlabel('Feature Importance')
ax1.set_title('Top-10 Feature Importance (All Indicators)')
ax1.invert_yaxis()


# 8.2 UDL indicators importance
# ax2 = axes[0, 1]
# bars2 = ax2.barh(range(len(udl_df)), udl_df['Mean_Importance'][::-1])
# ax2.set_yticks(range(len(udl_df)))
# ax2.set_yticklabels(udl_df['Feature'][::-1], fontsize=8)
# ax2.set_xlabel('Feature Importance')
# ax2.set_title('UDL Indicators Importance')
# ax2.invert_yaxis()

ax2 = axes[0, 1]
values2 = udl_df['Mean_Importance'][::-1]
# 使用正确的颜色映射调用方式
cmap = plt.get_cmap('jet')
norm2 = plt.Normalize(values2.min(), values2.max())
colors2 = cmap(norm2(values2))
bars2 = ax2.barh(range(len(udl_df)), values2,color=colors2)
ax2.set_yticks(range(len(udl_df)))
ax2.set_yticklabels(udl_df['Feature'][::-1], fontsize=8)
ax2.set_xlabel('Feature Importance')
ax2.set_title('UDL Indicators Importance')
ax2.invert_yaxis()

# 8.3 EEQ indicators importance
# ax3 = axes[0, 2]
# bars3 = ax3.barh(range(len(eeq_df)), eeq_df['Mean_Importance'][::-1])
# ax3.set_yticks(range(len(eeq_df)))
# ax3.set_yticklabels(eeq_df['Feature'][::-1], fontsize=8)
# ax3.set_xlabel('Feature Importance')
# ax3.set_title('EEQ Indicators Importance')
# ax3.invert_yaxis()

ax3 = axes[0, 2]
values3 = eeq_df['Mean_Importance'][::-1]
# 使用正确的颜色映射调用方式
cmap = plt.get_cmap('jet')
norm3 = plt.Normalize(values3.min(), values3.max())
colors3 = cmap(norm3(values3))
bars3 = ax3.barh(range(len(eeq_df)), values3,color=colors3)
ax3.set_yticks(range(len(eeq_df)))
ax3.set_yticklabels(eeq_df['Feature'][::-1], fontsize=8)
ax3.set_xlabel('Feature Importance')
ax3.set_title('EEQ Indicators Importance')
ax3.invert_yaxis()

# 8.4 Cumulative importance comparison
ax4 = axes[1, 0]
cumulative_all = all_df['Cumulative_Importance'].values
cumulative_udl = udl_df['Cumulative_Importance'].values
cumulative_eeq = eeq_df['Cumulative_Importance'].values

x_all = range(1, len(cumulative_all) + 1)
x_udl = range(1, len(cumulative_udl) + 1)
x_eeq = range(1, len(cumulative_eeq) + 1)

ax4.plot(x_all, cumulative_all, 'o-', label='All Indicators', linewidth=2, markersize=4)
ax4.plot(x_udl, cumulative_udl, 's-', label='UDL', linewidth=2, markersize=4)
ax4.plot(x_eeq, cumulative_eeq, '^-', label='EEQ', linewidth=2, markersize=4)
ax4.axhline(y=0.8, color='r', linestyle='--', alpha=0.7)
ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Cumulative Importance')
ax4.set_title('Cumulative Importance Comparison')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)

# 8.5 Core driving factors summary
ax5 = axes[1, 1]
# Find top 3 most important features for each group
top_features_summary = {
    'All Indicators': all_df.head(3)['Feature'].tolist(),
    'UDL': udl_df.head(3)['Feature'].tolist(),
    'EEQ': eeq_df.head(3)['Feature'].tolist()
}

# Create simple text summary
summary_text = "Core Driving Factors Summary:\n\n"
for group, features in top_features_summary.items():
    summary_text += f"{group}:\n"
    for i, feat in enumerate(features, 1):
        # Get importance value
        if group == 'All Indicators':
            importance = all_df[all_df['Feature'] == feat]['Mean_Importance'].values[0]
        elif group == 'UDL':
            importance = udl_df[udl_df['Feature'] == feat]['Mean_Importance'].values[0]
        else:
            importance = eeq_df[eeq_df['Feature'] == feat]['Mean_Importance'].values[0]
        summary_text += f"  {i}. {feat} ({importance:.4f})\n"
    summary_text += "\n"

ax5.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
ax5.axis('off')
# ax5.set_title('Core Driving Factors Summary')

# 8.6 CCD time series
ax6 = axes[1, 2]
years = df.index
ccd_values = y.values
ax6.plot(years, ccd_values, 'o-', linewidth=2, markersize=8, color='red')
ax6.fill_between(years, ccd_values, alpha=0.2, color='red')
ax6.set_xlabel('Year')
ax6.set_ylabel('CCD Value')
ax6.set_title('CCD Time Series')
ax6.grid(True, alpha=0.3)
ax6.set_xticks(years)

# plt.suptitle('CCD Core Driving Factors Analysis (Zhuhai)', fontsize=14, fontweight='bold')
# plt.tight_layout()

# Save the figure
plt.savefig('CCD_Core_Driving_Factors_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. Results output
print("\n" + "="*60)
print("Analysis Results Summary")
print("="*60)

# 9.1 All indicators Top-5
print("\n1. Top-5 Core Driving Factors (All Indicators):")
for i, row in all_df.head(5).iterrows():
    print(f"   {row['Rank']}. {row['Feature']} (Importance: {row['Mean_Importance']:.4f})")

# 9.2 UDL Top-3
print("\n2. Top-3 Core Driving Factors (UDL):")
for i, row in udl_df.head(3).iterrows():
    print(f"   {row['Rank']}. {row['Feature']} (Importance: {row['Mean_Importance']:.4f})")

# 9.3 EEQ Top-3
print("\n3. Top-3 Core Driving Factors (EEQ):")
for i, row in eeq_df.head(3).iterrows():
    print(f"   {row['Rank']}. {row['Feature']} (Importance: {row['Mean_Importance']:.4f})")

# 9.4 Features needed to explain 80% variance
print("\n4. Minimum Features to Explain 80% Variance:")
for df_group, name in [(all_df, "All Indicators"), (udl_df, "UDL"), (eeq_df, "EEQ")]:
    cumulative = df_group['Cumulative_Importance'].values
    n_features_80 = np.where(cumulative >= 0.8)[0]
    if len(n_features_80) > 0:
        n_features_80 = n_features_80[0] + 1
    else:
        n_features_80 = len(cumulative)
    
    percentage = n_features_80 / len(df_group) * 100
    print(f"   {name}: {n_features_80} features ({percentage:.1f}% of total)")

# 9.5 Correlation with original weights
print("\n5. Correlation with Original Weights:")

# Calculate correlation for each group
def calculate_weight_correlation(df_group):
    # Get original weights for features in this group
    original_weights = [weights[feat] for feat in df_group['Feature']]
    importance_values = df_group['Mean_Importance'].values
    correlation = np.corrcoef(original_weights, importance_values)[0, 1]
    return correlation

all_corr = calculate_weight_correlation(all_df)
udl_corr = calculate_weight_correlation(udl_df)
eeq_corr = calculate_weight_correlation(eeq_df)

print(f"   All Indicators: r = {all_corr:.3f}")
print(f"   UDL: r = {udl_corr:.3f}")
print(f"   EEQ: r = {eeq_corr:.3f}")

# 10. Save results
try:
    all_df.to_csv('All_Indicators_Feature_Importance.csv', index=False)
    udl_df.to_csv('UDL_Feature_Importance.csv', index=False)
    eeq_df.to_csv('EEQ_Feature_Importance.csv', index=False)
    
    print("\n" + "="*60)
    print("Results Saved:")
    print("  1. All_Indicators_Feature_Importance.csv")
    print("  2. UDL_Feature_Importance.csv")
    print("  3. EEQ_Feature_Importance.csv")
    print("  4. CCD_Core_Driving_Factors_Analysis.png")
    print("="*60)
except Exception as e:
    print(f"\nError saving files: {str(e)}")

# 11. Additional analysis: CCD correlation with each indicator
print("\n" + "="*60)
print("CCD Correlation with Each Indicator")
print("="*60)

# Calculate correlation coefficients
correlations_with_ccd = {}
for feature in all_features:
    correlations_with_ccd[feature] = np.corrcoef(df[feature], y)[0, 1]

# Create correlation DataFrame
corr_df = pd.DataFrame(list(correlations_with_ccd.items()), columns=['Feature', 'Correlation_with_CCD'])
corr_df = corr_df.sort_values('Correlation_with_CCD', ascending=False)
corr_df['Rank'] = range(1, len(corr_df) + 1)

print("\nTop-5 Positive Correlations with CCD:")
for i, row in corr_df.head(5).iterrows():
    print(f"  {row['Rank']}. {row['Feature']}: {row['Correlation_with_CCD']:.4f}")

print("\nTop-5 Negative Correlations with CCD:")
for i, row in corr_df.tail(5).iterrows():
    print(f"  {row['Rank']}. {row['Feature']}: {row['Correlation_with_CCD']:.4f}")

# Save correlation results
corr_df.to_csv('CCD_Correlation_Analysis.csv', index=False)
print("\nCCD correlation analysis saved as: CCD_Correlation_Analysis.csv")

print("\n✓ Analysis completed successfully!")