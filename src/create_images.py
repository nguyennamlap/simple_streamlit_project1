import pandas as pd
import numpy as np
import os
from handle.eda import *

path_before = '../images/before_handle/'
path_mid = '../images/before_train/'
path_final = '../images/after_train/'

def create_images(df: pd.DataFrame, path_stage: str, target_col: str = 'TARGET', 
                  key_numerical_features: list = None, stage_name: str = ""):
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(path_stage, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🎨 BẮT ĐẦU TẠO BIỂU ĐỒ - {stage_name}")
    print("="*80)
    print(f"📂 Thư mục lưu: {path_stage}")
    print(f"📊 Kích thước dữ liệu: {df.shape}")
    
    # Mặc định các feature số quan trọng nếu không được cung cấp
    if key_numerical_features is None:
        key_numerical_features = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
            'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
        ]
        # Thêm các cột được tạo nếu có
        engineered_cols = ['Tỉ lệ vay so với nhu cầu', 'OWN_CAR_AGE']
        key_numerical_features.extend([col for col in engineered_cols if col in df.columns])
    
    try:
        # 1. Phân tích thông tin cơ bản
        print("\n📋 1. Phân tích thông tin cơ bản...")
        basic_info(df)
        
        # 2. Phân tích giá trị thiếu
        print("\n🔍 2. Phân tích giá trị thiếu...")
        missing_values_analysis(df,path=path_stage)
        
        # 3. Phân tích target (nếu có)
        if target_col in df.columns:
            print(f"\n🎯 3. Phân tích biến mục tiêu '{target_col}'...")
            
            # Biểu đồ cột (bar chart)
            target_analysis_plot(df, target_col, path_stage)
            
            # Biểu đồ tròn (pie chart)
            target_analysis_pie(df, target_col, path_stage)
        else:
            print(f"\n⚠️  Cột target '{target_col}' không tồn tại, bỏ qua phân tích target")
        
        # 4. Phân tích các đặc trưng số
        print("\n📊 4. Phân tích các đặc trưng số...")
        numerical_features_analysis(df, target_col, path_stage, key_numerical_features)
        
        # 5. Phân tích các đặc trưng phân loại
        print("\n📂 5. Phân tích các đặc trưng phân loại...")
        categorical_features_analysis(df, target_col, path_stage)
        
        # 6. Phân tích tương quan với target (categorical)
        if target_col in df.columns:
            print(f"\n🔗 6. Phân tích mối quan hệ categorical với '{target_col}'...")
            categorical_target_relationship(df, target_col)
        
        # 7. Báo cáo tổng quan categorical
        print("\n📋 7. Báo cáo tổng quan đặc trưng phân loại...")
        bao_cao_tong_quan_categorical(df, target_col)
        
        # 8. Phân tích tương quan (correlation)
        print("\n🔥 8. Phân tích ma trận tương quan...")
        correlation_analysis(df, target_col, path_stage)
        
        # 9. Phân tích các features được tạo (nếu có)
        engineered_features = ['Tỉ lệ vay so với nhu cầu', 'Sở hữu xe', 
                              'OCCUPATION_TYPE_ENHANCED', 'IS_RETIRED_NO_OCCUPATION', 
                              'IS_WORKING_NO_OCCUPATION']
        
        has_engineered = any(feat in df.columns for feat in engineered_features)
        if has_engineered:
            print("\n⚙️  9. Phân tích các features được tạo...")
            engineered_features_analysis(df)
        
        # 10. Báo cáo tổng quan
        print("\n📝 10. Tạo báo cáo tổng quan...")
        generate_summary_report(df)
        
        print("\n" + "="*80)
        print(f"✅ HOÀN THÀNH TẠO BIỂU ĐỒ - {stage_name}")
        print("="*80)
        print(f"📂 Tất cả biểu đồ đã được lưu tại: {path_stage}")
        print("\n📁 Danh sách file đã tạo:")
        
        # Liệt kê các file đã tạo
        expected_files = [
            'missing_values.png',
            f'{target_col}_bar_distribution.png',
            f'{target_col}_pie_distribution.png',
            'numerical_distributions.png',
            'top_10.png',
            'correlation_heatmap.png'
        ]
        
        for file in expected_files:
            file_path = os.path.join(path_stage, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"   ✓ {file} ({file_size:.1f} KB)")
            else:
                print(f"   ✗ {file} (không tồn tại)")
        
    except Exception as e:
        print(f"\n❌ LỖI KHI TẠO BIỂU ĐỒ: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
 
    print("="*80)
    print("🚀 BẮT ĐẦU QUÁ TRÌNH EDA CHO TẤT CẢ CÁC GIAI ĐOẠN")
    print("="*80)
    key_features_before = [
    # Tài chính
    'AMT_INCOME_TOTAL',
    'AMT_CREDIT',
    'AMT_GOODS_PRICE',
    'AMT_ANNUITY',

    # Nhân khẩu học
    'DAYS_BIRTH',
    'DAYS_EMPLOYED',
    'OWN_CAR_AGE',

    # Nguồn external score (rất quan trọng)
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3'
]

    # ==================== GIAI ĐOẠN 1: BEFORE HANDLE ====================
    print("\n" + "🔵" * 40)
    print("GIAI ĐOẠN 1: DỮ LIỆU TRƯỚC KHI XỬ LÝ")
    print("🔵" * 40)
    
    try:
        df_before = pd.read_csv('../data/application_train.csv')
        create_images(
            df=df_before,
            path_stage=path_before,
            target_col='TARGET',
            key_numerical_features=key_features_before,
            stage_name="GIAI ĐOẠN 1: BEFORE HANDLE"
        )
    except FileNotFoundError:
        print("⚠️  Không tìm thấy file 'data/application_train.csv'")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý giai đoạn BEFORE HANDLE: {str(e)}")
    
    # ==================== GIAI ĐOẠN 2: BEFORE TRAIN ====================
    print("\n" + "🟡" * 40)
    print("GIAI ĐOẠN 2: DỮ LIỆU SAU KHI XỬ LÝ - TRƯỚC KHI TRAIN")
    print("🟡" * 40)
    
    try:
        df_mid = pd.read_csv('../data/df_processed.csv')  # Hoặc file đã xử lý của bạn
        
        # Có thể thêm các features được tạo vào danh sách
        key_features_mid = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
            'Tỉ lệ vay so với nhu cầu',  
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'OWN_CAR_AGE'
        ]
        
        create_images(
            df=df_mid,
            path_stage=path_mid,
            target_col='TARGET',
            key_numerical_features=key_features_mid,
            stage_name="GIAI ĐOẠN 2: BEFORE TRAIN"
        )
    except FileNotFoundError:
        print("⚠️  Không tìm thấy file dữ liệu đã xử lý")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý giai đoạn BEFORE TRAIN: {str(e)}")
    
    # ==================== GIAI ĐOẠN 3: AFTER TRAIN ====================
    print("\n" + "🟢" * 40)
    print("GIAI ĐOẠN 3: DỮ LIỆU SAU KHI TRAIN")
    print("🟢" * 40)
    
    try:
        df_final = pd.read_csv('../data/df_final.csv')  # File cuối cùng từ code xử lý
        
        # Bao gồm tất cả các features đã được tạo
        key_features_final = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
            'Tỉ lệ vay so với nhu cầu',
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'EXT_SOURCE_1_is_missing', 'EXT_SOURCE_2_is_missing', 'EXT_SOURCE_3_is_missing',
            'IS_RETIRED_NO_OCCUPATION', 'IS_WORKING_NO_OCCUPATION'
        ]
        
        create_images(
            df=df_final,
            path_stage=path_final,
            target_col='TARGET',
            key_numerical_features=key_features_final,
            stage_name="GIAI ĐOẠN 3: AFTER TRAIN"
        )
    except FileNotFoundError:
        print("⚠️  Không tìm thấy file 'df_final.csv'")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý giai đoạn AFTER TRAIN: {str(e)}")
    
    # ==================== HOÀN THÀNH ====================
    print("\n" + "="*80)
    print("🎉 HOÀN THÀNH TẤT CẢ CÁC GIAI ĐOẠN EDA")
    print("="*80)
    print("\n📊 Tổng kết:")
    print(f"   • Giai đoạn 1 (Before Handle): {path_before}")
    print(f"   • Giai đoạn 2 (Before Train):  {path_mid}")
    print(f"   • Giai đoạn 3 (After Train):   {path_final}")
    print("\n💡 Kiểm tra các thư mục trên để xem kết quả phân tích!")


if __name__ == "__main__":
    main()