import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(file_path='data/application_train.csv'):
    """Load the processed data"""
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def basic_info(df):
    """Display basic information about the dataset"""
    print("\n" + "="*80)
    print("BASIC INFORMATION")
    print("="*80)
    
    print("\nSố cột và hàng", df.shape)
    print("\nKiểu dữ liệu của các cột:")
    print(df.dtypes.value_counts())
    
    print("\nTHông tin tổng quan về dữ liệu 5 dòng đầu:")
    print(df.head())

def missing_values_analysis(df,path):
    """Analyze missing values in the dataset"""
    print("\n" + "="*80)
    print("MISSING VALUES ANALYSIS")
    print("="*80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100 # tỉ lệ thiếu dữ liệu theo phần trăm
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    # sắp xếp theo tỉ lệ thiếu dữ liệu giảm dần và lọc chỉ những cột có thiếu dữ liệu
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    )
    # Hiển thị bảng thiếu dữ liệu
    if len(missing_df) > 0:
        print("\nColumns with missing values:")
        # chuyển đổi hiển thị đầy đủ bảng không bị cắt bớt khi in ra
        print(missing_df.to_string(index=False))
        
        # Plot missing values
        plt.figure(figsize=(12, 6))
        top_missing = missing_df.head(20)
        plt.barh(top_missing['Column'], top_missing['Missing_Percentage'], color='salmon', edgecolor='black', alpha=0.7)
        plt.xlabel('Missing Percentage (%)')
        plt.title('Top 20 cột dữ liệu bị thiếu nhiều nhất')
        plt.tight_layout()
        plt.savefig(f'{path}/missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nĐã lưu biểu đồ thiếu dữ liệu vào 'images/before_train/missing_values.png'")
    else:
        print("\nNo missing values found!")

def target_analysis_plot(df, target_col, path):
    print("\n" + "="*80)
    print(f"Cột được phân tích là: ({target_col})")
    print("="*80)

    if target_col not in df.columns:
        print(f"\nTarget column '{target_col}' not found in dataset")
        return

    # Thống kê
    print("\nCác giá trị của cột mục tiêu:")
    print(df[target_col].value_counts())

    print("\nTỷ lệ các giá trị trong cột mục tiêu:")
    print(df[target_col].value_counts(normalize=True))

    # Chuẩn bị dữ liệu
    counts = df[target_col].value_counts()
    colors = ['#27ae60', '#c0392b']

    # Tạo figure & axis (1 axis duy nhất)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Vẽ biểu đồ
    counts.plot(
        kind='bar',
        ax=ax,
        color=colors,
        edgecolor='black',
        alpha=0.9,
        width=0.7
    )

    # Trang trí biểu đồ
    ax.set_title(
        f'Phân Bố Biến Mục Tiêu của cột {target_col}',
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax.set_xlabel(f'Các Lớp Mục Tiêu của cột {target_col}', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_xticklabels(['Không Vỡ Nợ (0)', 'Vỡ Nợ (1)'], rotation=0)

    # Ghi số lượng trên cột
    for i, v in enumerate(counts): # i: vị trí cột, v: giá trị cột
        ax.text(
            i,
            v + max(counts) * 0.01,
            f'{v:,}', # định dạng số có dấu phẩy
            ha='center',
            fontweight='bold' # căn giữa và in đậm
        )

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # lưới ngang chỉ theo trục y
    # Lưu & đóng
    plt.tight_layout()
    plt.savefig(f"{path}/{target_col}_bar_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def target_analysis_pie(df, target_col, path):
    print("\n" + "="*80)
    print(f"Cột được phân tích là: ({target_col})")
    print("="*80)

    if target_col not in df.columns:
        print(f"\nTarget column '{target_col}' not found in dataset")
        return

    # Thống kê
    print("\nCác giá trị của cột mục tiêu:")
    print(df[target_col].value_counts())

    print("\nTỷ lệ các giá trị trong cột mục tiêu:")
    print(df[target_col].value_counts(normalize=True))
    
    # Lấy số lượng giá trị duy nhất
    n_unique = df[target_col].nunique()
    
    # Tạo màu sắc dựa trên số lượng giá trị
    if n_unique == 2:
        colors = ['#27ae60', '#c0392b']  # Xanh lá, Đỏ
    elif n_unique == 3:
        colors = ['#27ae60', '#f39c12', '#c0392b']  # Xanh lá, Cam, Đỏ
    elif n_unique == 4:
        colors = ['#27ae60', '#3498db', '#f39c12', '#c0392b']  # Xanh lá, Xanh dương, Cam, Đỏ
    elif n_unique == 5:
        colors = ['#27ae60', '#3498db', '#f1c40f', '#f39c12', '#c0392b']  # Xanh lá, Xanh dương, Vàng, Cam, Đỏ
    else:
        # Sử dụng colormap cho nhiều giá trị hơn
        cmap = plt.cm.Set3  # Hoặc plt.cm.tab20, plt.cm.Set2
        colors = [cmap(i) for i in np.linspace(0, 1, n_unique)]
    
    # Tính toán kích thước figure dựa trên số lượng giá trị
    if n_unique <= 5:
        figsize = (8, 6)
    elif n_unique <= 8:
        figsize = (10, 7)
    else:
        figsize = (12, 8)
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Lấy giá trị và nhãn
    value_counts = df[target_col].value_counts()
    labels = [f"{idx} ({val})" for idx, val in value_counts.items()]
    
    # Pie chart với nhiều tùy chọn
    wedges, texts, autotexts = ax.pie(value_counts, 
                                       labels=labels if n_unique <= 8 else None,
                                       autopct='%1.1f%%', 
                                       colors=colors, 
                                       startangle=90, 
                                       counterclock=False,
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'alpha': 0.9},
                                       textprops={'fontsize': 10})
    
    # Định dạng autopct (phần trăm)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    # Tiêu đề
    ax.set_title(f'Phân Phối Cột {target_col}', fontsize=14, fontweight='bold', pad=20)
    
    # Thêm chú thích nếu có nhiều giá trị
    if n_unique > 8:
        # Tạo legend với layout thích hợp
        n_cols = 2 if n_unique > 12 else 1
        ax.legend(wedges, labels, 
                  title=f"Giá trị ({n_unique} loại)",
                  loc="center left", 
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9,
                  ncol=n_cols)
    
    # Thêm tổng số lượng
    total = value_counts.sum()
    ax.text(0, -1.2, f'Tổng số: {total:,}', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f"{path}/{target_col}_pie_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nBiểu đồ phân phối cột '{target_col}' đã được lưu vào {path}")

def numerical_features_analysis(df: pd.DataFrame, target_col: str, path, key_features: list):
   
    print("\n" + "="*80)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*80)
    
    # chọn cột có dạng chữ, lấy tên cột, bỏ vào list python 
    # categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # chọn cột có dạng số, lấy tên cột, bỏ vào list python 
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # loại bỏ cột mục tiêu khỏi danh sách phân tích nếu có

    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    print(f"\nNumber of numerical features: {len(numerical_cols)}")
    print("\nNumerical Features Statistics:")
    print(df[numerical_cols].describe())
    
    # Analyze key features
    # key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 
    #                 'Tỉ lệ vay so với nhu cầu', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    available_features = [f for f in key_features if f in df.columns]
    
    if available_features:
        print(f"\nKey Features Analysis:")
        for feature in available_features:
            print(f"\n{feature}:")
            print(f"  Mean: {df[feature].mean():.2f}")
            print(f"  Median: {df[feature].median():.2f}")
            print(f"  Std: {df[feature].std():.2f}")
            print(f"  Min: {df[feature].min():.2f}")
            print(f"  Max: {df[feature].max():.2f}")
        
        # Plot distributions
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(15, 4.5 * n_rows),
                                constrained_layout=True)

        # Flatten axes
        axes = axes.flatten() if n_features > 1 else [axes]

        # Tạo màu gradient đẹp
        cmap = plt.cm.coolwarm
        colors = [cmap(i) for i in np.linspace(0.2, 0.8, n_features)]

        for idx, feature in enumerate(available_features):
            if idx < len(axes):
                ax = axes[idx]
                data = df[feature].dropna()
                
                if len(data) == 0:
                    ax.text(0.5, 0.5, 'Không có dữ liệu', 
                        ha='center', va='center', fontsize=11, color='gray')
                    ax.set_title(f'{feature}\n(No Data)', fontsize=10, color='gray')
                    continue
                
                # Tính toán bins tối ưu
                q75, q25 = np.percentile(data, [75, 25])
                iqr = q75 - q25
                bin_width = 2 * iqr / (len(data) ** (1/3))
                n_bins = int((data.max() - data.min()) / bin_width) if bin_width > 0 else 50
                
                # Vẽ histogram
                color = colors[idx % len(colors)]
                n, bins, patches = ax.hist(data, 
                                        bins=min(n_bins, 50),
                                        color=color,
                                        edgecolor='white',
                                        linewidth=1.5,
                                        alpha=0.85,
                                        density=False)
                
                # Thêm thông tin thống kê
                mean_val = data.mean()
                median_val = data.median()
                
                ax.axvline(mean_val, color='#e74c3c', linestyle='--', 
                        linewidth=2, alpha=0.8, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='#2ecc71', linestyle='-', 
                        linewidth=2, alpha=0.6, label=f'Median: {median_val:.2f}')
                
                # Định dạng đẹp
                ax.set_title(f'Phân phối: {feature}', 
                            fontsize=12, fontweight='bold', pad=10)
                ax.set_xlabel('Giá trị', fontsize=10)
                ax.set_ylabel('Tần suất', fontsize=10)
                ax.grid(axis='y', alpha=0.2, linestyle='--')
                
                # Thêm text box thống kê
                stats_text = f'N={len(data):,}\nMean={mean_val:.2f}\nStd={data.std():.2f}'
                ax.text(0.97, 0.97, stats_text, 
                    transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Chỉ thêm legend nếu cần
                if idx < 3:  # Chỉ 3 biểu đồ đầu
                    ax.legend(loc='upper right', fontsize=8)

        # Ẩn các subplot không sử dụng
        for idx in range(len(available_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{path}/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nNumerical distributions plot saved as 'numerical_distributions.png'")

def categorical_features_analysis(df, target_col, path):
    
    # Header với format đẹp
    print("\n" + "═" * 80)
    print("📊 PHÂN TÍCH ĐẶC TRƯNG PHÂN LOẠI")
    print("═" * 80)
    
    # Xác định các cột phân loại
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Nếu có target_col và là categorical, loại bỏ nó
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    print(f"\n🔍 Số lượng đặc trưng phân loại: {len(categorical_cols)}")

    if len(categorical_cols) == 0:
        print("❌ Không tìm thấy đặc trưng phân loại nào")
        return
    
    # Danh sách các feature quan trọng cần phân tích
    key_cat_features = ['Sở hữu xe', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE_ENHANCED', 
                        'OCCUPATION_MISSING_TYPE', 'NAME_INCOME_TYPE', 'NAME_CONTRACT_TYPE']
    
    # Chỉ lấy những feature có trong dataframe
    available_cat = [f for f in key_cat_features if f in df.columns]
    
    # Nếu không có feature trong danh sách key, lấy tất cả categorical features
    if not available_cat:
        available_cat = categorical_cols[:6]  # Lấy tối đa 6 features đầu tiên
    
    print(f"\n✨ Phân tích {len(available_cat)} đặc trưng quan trọng:")
    for i, feature in enumerate(available_cat, 1):
        unique_count = df[feature].nunique()
        missing_count = df[feature].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        print(f"   {i:2d}. {feature:30s} | Giá trị duy nhất: {unique_count:3d} | "
              f"Thiếu: {missing_count:5,d} ({missing_pct:5.1f}%)")
    
    # PHẦN 1: THỐNG KÊ CHI TIẾT
    print("\n" + "─" * 80)
    print("📈 THỐNG KÊ CHI TIẾT TỪNG ĐẶC TRƯNG")
    print("─" * 80)
    
    for feature in available_cat[:5]:  # Hiển thị chi tiết 5 features đầu
        print(f"\n📋 {feature}:")
        print("-" * 40)
        
        value_counts = df[feature].value_counts(dropna=False)
        value_counts_pct = df[feature].value_counts(normalize=True, dropna=False) * 100
        
        # Hiển thị top 10 giá trị
        for i, (value, count) in enumerate(value_counts.head(10).items(), 1):
            pct = value_counts_pct.get(value, 0)
            if pd.isna(value):
                value_str = "NULL/MISSING"
            else:
                value_str = str(value)
            print(f"   {i:2d}. {value_str:30s}: {count:7,d} ({pct:5.1f}%)")
        
        # Thông tin tổng quan
        print(f"   Tổng số giá trị duy nhất: {value_counts.shape[0]}")
        if value_counts.shape[0] > 10:
            print(f"   ... và {value_counts.shape[0] - 10} giá trị khác")
    
    # PHẦN 2: VẼ BIỂU ĐỒ
    print("\n" + "─" * 80)
    print("🎨 VẼ BIỂU ĐỒ PHÂN PHỐI")
    print("─" * 80)
    
    n_features = len(available_cat)
    n_cols = 2
    n_rows = min(3, (n_features + n_cols - 1) // n_cols)
    
    # Tạo figure với layout đẹp
    fig = plt.figure(figsize=(18, 6 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.3, wspace=0.2)
    
    for idx, feature in enumerate(available_cat[:n_rows * 2]):  # Tối đa 6 biểu đồ
        if idx >= n_rows * 2:
            break
            
        row = idx // 2
        col = idx % 2
        
        ax1 = fig.add_subplot(gs[row, col])
        
        # Chuẩn bị dữ liệu
        value_counts = df[feature].value_counts().head(10)
        values = value_counts.index.tolist()
        counts = value_counts.values.tolist()
        
        # Tạo màu gradient
        if len(counts) > 0:
            colors = plt.cm.Set3(np.linspace(0.2, 0.8, len(counts)))
        else:
            colors = ['#3498db']
        
        # Vẽ horizontal bar chart
        bars = ax1.barh(range(len(counts)), counts, color=colors, edgecolor='white', height=0.7)
        
        # Đảo ngược trục y để giá trị lớn nhất ở trên
        ax1.set_yticks(range(len(counts)))
        ax1.set_yticklabels([str(v)[:30] + ('...' if len(str(v)) > 30 else '') for v in values])
        
        # Thêm số liệu trên mỗi cột
        total = len(df)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100
            ax1.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,} ({percentage:.1f}%)', 
                    va='center', fontsize=9, fontweight='bold')
        
        # Định dạng biểu đồ
        ax1.set_xlabel('Số lượng', fontsize=10)
        ax1.set_title(f'📊 {feature}\nTop {len(counts)} giá trị phổ biến', 
                     fontsize=12, fontweight='bold', pad=12)
        
        # Thêm grid nhẹ
        ax1.grid(axis='x', alpha=0.2, linestyle='--')
        
        # Xóa khung không cần thiết
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Thêm thông tin tổng quan
        unique_count = df[feature].nunique()
        missing_count = df[feature].isnull().sum()
        ax1.text(0.02, 0.98, 
                f'Giá trị duy nhất: {unique_count}\nThiếu: {missing_count:,}',
                transform=ax1.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Tiêu đề tổng
    fig.suptitle('PHÂN TÍCH PHÂN PHỐI ĐẶC TRƯNG PHÂN LOẠI', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Lưu biểu đồ nếu có path
    if path:
        # Đảm bảo thư mục tồn tại
        plt.savefig(f'{path}/top_10.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Biểu đồ đã được lưu tại: {path}")
    
    
def categorical_target_relationship(df: pd.DataFrame, target_col: str):
    print("\n" + "─" * 80)
    print(f"🎯 PHÂN TÍCH MỐI QUAN HỆ VỚI TARGET: {target_col}")
    print("─" * 80)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    available_cat = [col for col in categorical_cols if col != target_col]
    # Chọn 3 features quan trọng nhất để phân tích với target
    top_features = available_cat[:3]
    
    if top_features:
        fig_target, axes_target = plt.subplots(1, min(3, len(top_features)), 
                                                figsize=(5 * min(3, len(top_features)), 6))
        
        if len(top_features) == 1:
            axes_target = [axes_target]
        
        for idx, feature in enumerate(top_features):
            ax = axes_target[idx] if len(top_features) > 1 else axes_target
            
            # Tạo crosstab với target
            crosstab = pd.crosstab(df[feature].fillna('MISSING'), 
                                    df[target_col], 
                                    normalize='index') * 100
            
            # Vẽ stacked bar chart
            crosstab.plot(kind='bar', ax=ax, stacked=True, 
                            color=['#2ecc71', '#e74c3c'], 
                            edgecolor='black', alpha=0.85)
            
            # Định dạng
            ax.set_title(f'{feature} vs {target_col}', fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Tỷ lệ (%)', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title=target_col, labels=['Class 0', 'Class 1'])
            
            # Thêm tổng số mẫu trên mỗi nhóm
            totals = df[feature].fillna('MISSING').value_counts()
            for i, total in enumerate(totals):
                ax.text(i, 102, f'n={total}', ha='center', fontsize=8)
        
        plt.suptitle(f'PHÂN TÍCH TƯƠNG QUAN VỚI BIẾN MỤC TIÊU: {target_col}', 
                    fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        plt.show()
        
        # In thông tin chi tiết về mối quan hệ với target
        print(f"\n📌 Tỷ lệ mục tiêu theo từng nhóm:")
        for feature in top_features[:2]:  # Chỉ phân tích 2 features
            print(f"\n   {feature}:")
            crosstab_counts = pd.crosstab(df[feature].fillna('MISSING'), df[target_col])
            crosstab_pct = pd.crosstab(df[feature].fillna('MISSING'), 
                                        df[target_col], normalize='index')
            
            for category in crosstab_counts.index[:5]:  # Hiển thị top 5 categories
                count_0 = crosstab_counts.loc[category, 0]
                count_1 = crosstab_counts.loc[category, 1]
                pct_1 = crosstab_pct.loc[category, 1] * 100
                print(f"      • {category[:20]:20s}: "
                        f"Class 0: {count_0:5,d} | Class 1: {count_1:5,d} "
                        f"({pct_1:5.1f}% default)")
    
def bao_cao_tong_quan_categorical(df, target_col='TARGET'):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print("\n" + "═" * 80)
    print("📋 BÁO CÁO TỔNG QUAN ĐẶC TRƯNG PHÂN LOẠI")
    print("═" * 80)
    
    # Tạo dataframe tổng quan
    summary_data = []
    for feature in categorical_cols:
        unique_count = df[feature].nunique()
        missing_count = df[feature].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        vc = df[feature].value_counts(dropna=True)
        most_common = vc.index[0] if not vc.empty else "N/A"
        most_common_pct = (df[feature].value_counts().iloc[0] / len(df)) * 100 if unique_count > 0 else 0
        available_cat = [col for col in categorical_cols if col != target_col]
        summary_data.append({
            'Feature': feature,
            'Unique Values': unique_count,
            'Missing': f"{missing_count:,} ({missing_pct:.1f}%)",
            'Most Common': f"{most_common} ({most_common_pct:.1f}%)",
            'In Key Features': '✓' if feature in available_cat else ''
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(f"\nTổng số đặc trưng phân loại: {len(summary_df)}")
    print(f"\n{summary_df.to_string(index=False)}")
    
    # Khuyến nghị
    print("\n" + "💡 KHUYẾN NGHỊ:")
    print("-" * 40)
    
    high_cardinality = [f for f in categorical_cols if df[f].nunique() > 50]
    if high_cardinality:
        print(f"⚠️  Đặc trưng có cardinality cao (>50): {', '.join(high_cardinality)}")
        print("   → Xem xét: Grouping, Target Encoding, hoặc bỏ qua")
    
    high_missing = [f for f in categorical_cols if df[f].isnull().mean() > 0.3]
    if high_missing:
        print(f"⚠️  Đặc trưng có nhiều missing (>30%): {', '.join(high_missing)}")
        print("   → Xem xét: Imputation hoặc loại bỏ")
    
    low_cardinality = [f for f in categorical_cols if df[f].nunique() == 2]
    if low_cardinality:
        print(f"✅ Đặc trưng binary tốt cho encoding: {', '.join(low_cardinality)}")
        print("   → Có thể dùng Label Encoding")
    
    print("\n" + "✅ PHÂN TÍCH HOÀN TẤT!")

def correlation_analysis(df, target_col='TARGET', path=None):
   
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        
        # Show top correlations with target
        if target_col in numerical_cols:
            target_corr = corr_matrix[target_col].sort_values(ascending=False)
            print(f"\nTop 15 Features Correlated with {target_col}:")
            print(target_corr.head(15))
        
        # Plot correlation heatmap for key features
        key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
                       'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
                       'IS_RETIRED_NO_OCCUPATION', 'IS_WORKING_NO_OCCUPATION']
        if target_col in df.columns:
            key_features.append(target_col)
        
        available_features = [f for f in key_features if f in numerical_cols]
        
        if len(available_features) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df[available_features].corr(), annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, linewidths=1)
            plt.title('Correlation Heatmap of Key Features')
            plt.tight_layout()
            plt.savefig(f'{path}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\nCorrelation heatmap saved as 'correlation_heatmap.png'")

def engineered_features_analysis(df):
    """Analyze the engineered features"""
    print("\n" + "="*80)
    print("ENGINEERED FEATURES ANALYSIS")
    print("="*80)
    
    engineered_features = {
        'Tỉ lệ vay so với nhu cầu': 'Credit to Goods Price Ratio',
        'Sở hữu xe': 'Car Ownership',
        'OCCUPATION_TYPE_ENHANCED': 'Enhanced Occupation Type',
        'IS_RETIRED_NO_OCCUPATION': 'Retired without Occupation Flag',
        'IS_WORKING_NO_OCCUPATION': 'Working without Occupation Flag'
    }
    
    available_eng = {k: v for k, v in engineered_features.items() if k in df.columns}
    
    print(f"\nEngineered Features Analysis:")
    for feature, description in available_eng.items():
        print(f"\n{feature} ({description}):")
        if df[feature].dtype == 'object':
            print(df[feature].value_counts())
        else:
            print(f"  Mean: {df[feature].mean():.4f}")
            print(f"  Median: {df[feature].median():.4f}")
            print(f"  Std: {df[feature].std():.4f}")

def generate_summary_report(df):
    """Generate a summary report"""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Features: {df.shape[1]}")
    print(f"Total Samples: {df.shape[0]}")
    print(f"\nNumerical Features: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"Categorical Features: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")



