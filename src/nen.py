"""
File CSV Compression Script - IN-PLACE COMPRESSION
Tự động nén file CSV lớn xuống kích thước nhỏ hơn BẰNG CÁCH THAY THẾ FILE GỐC
"""
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
import psutil
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')

class CSVCompressor:
    def __init__(self, target_size_mb=25):
        """
        Khởi tạo compressor
        
        Args:
            target_size_mb: Kích thước mục tiêu (MB)
        """
        self.target_size_mb = target_size_mb
        self.original_size_mb = None
        self.compressed_size_mb = None
        self.compression_ratio = None
        
    def optimize_dtypes(self, df):
        """
        Tối ưu hóa kiểu dữ liệu để giảm kích thước
        
        Args:
            df: DataFrame cần tối ưu
            
        Returns:
            DataFrame đã được tối ưu
        """
        print("🔍 Đang tối ưu kiểu dữ liệu...")
        
        # Sao chép DataFrame để không ảnh hưởng đến dữ liệu gốc
        df_optimized = df.copy()
        
        # Duyệt qua từng cột
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            # Tối ưu kiểu số nguyên
            if col_type in ['int64', 'int32']:
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                # Kiểm tra xem có giá trị NaN không
                has_nan = df_optimized[col].isna().any()
                
                if not has_nan:
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    # Nếu có NaN, chuyển sang kiểu float tối ưu
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df_optimized[col] = df_optimized[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
            
            # Tối ưu kiểu số thực
            elif col_type in ['float64', 'float32']:
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                # Giảm độ chính xác float64 -> float32 -> float16
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df_optimized[col] = df_optimized[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
            
            # Tối ưu kiểu object/string
            elif col_type == 'object':
                # Kiểm tra xem có phải là string không
                if df_optimized[col].apply(lambda x: isinstance(x, str)).all():
                    # Chuyển sang kiểu category nếu số lượng unique nhỏ
                    num_unique = df_optimized[col].nunique()
                    num_total = len(df_optimized[col])
                    
                    if num_unique / num_total < 0.5:  # Nếu ít hơn 50% unique
                        df_optimized[col] = df_optimized[col].astype('category')
        
        # Tính toán mức tiết kiệm bộ nhớ
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2  # MB
        savings = ((original_memory - optimized_memory) / original_memory) * 100
        
        print(f"✅ Tối ưu hoàn tất:")
        print(f"   - Trước: {original_memory:.2f} MB")
        print(f"   - Sau: {optimized_memory:.2f} MB")
        print(f"   - Tiết kiệm: {savings:.1f}%")
        
        return df_optimized
    
    def compress_csv_in_place(self, input_path, backup_original=True):
        """
        Nén file CSV NGAY TẠI CHỖ (in-place) bằng cách lưu lại dưới dạng CSV đã nén
        
        Args:
            input_path: Đường dẫn file CSV gốc
            backup_original: Có tạo backup file gốc không
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        print(f"\n📁 Đang xử lý file: {input_path}")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(input_path):
            print(f"❌ File không tồn tại: {input_path}")
            return False
        
        # Tạo đường dẫn tạm thời
        temp_path = f"{input_path}.temp_compressed"
        
        # Tính kích thước file gốc
        self.original_size_mb = os.path.getsize(input_path) / (1024**2)
        print(f"📊 Kích thước gốc: {self.original_size_mb:.2f} MB")
        
        try:
            # Tạo backup nếu cần
            if backup_original:
                backup_path = f"{input_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(input_path, backup_path)
                print(f"💾 Đã tạo backup: {backup_path}")
            
            # Đọc file CSV với chunksize để xử lý file lớn
            print("📖 Đang đọc file CSV...")
            chunksize = 100000  # Số dòng mỗi chunk
            chunks = []
            
            for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
                chunks.append(chunk)
                print(f"   Đã đọc chunk {i+1}: {len(chunk):,} dòng", end='\r')
            
            print(f"\n✅ Đã đọc toàn bộ file: {sum(len(c) for c in chunks):,} dòng")
            
            # Kết hợp tất cả chunks
            df = pd.concat(chunks, ignore_index=True)
            
            # Tối ưu kiểu dữ liệu
            df_optimized = self.optimize_dtypes(df)
            
            # Nếu vẫn quá lớn, giảm độ chính xác thêm
            estimated_size = df_optimized.memory_usage(deep=True).sum() / (1024**2)
            
            if estimated_size > self.target_size_mb * 1.5:
                print("📉 Kích thước vẫn lớn, đang giảm độ chính xác thêm...")
                # Giảm tất cả float64 -> float32
                for col in df_optimized.columns:
                    if df_optimized[col].dtype == 'float64':
                        df_optimized[col] = df_optimized[col].astype('float32')
                    elif df_optimized[col].dtype == 'int64':
                        df_optimized[col] = df_optimized[col].astype('int32')
            
            # Lưu thành CSV đã nén (sử dụng compression)
            print("💾 Đang lưu file CSV đã nén...")
            
            # Lưu với định dạng CSV nén gzip
            df_optimized.to_csv(
                temp_path,
                index=False,
                compression='gzip'  # Nén gzip cho CSV
            )
            
            # Kiểm tra kích thước file tạm
            temp_size_mb = os.path.getsize(temp_path) / (1024**2)
            print(f"📊 Kích thước file tạm: {temp_size_mb:.2f} MB")
            
            # Nếu file tạm nhỏ hơn file gốc, thay thế file gốc
            if temp_size_mb < self.original_size_mb:
                # Xóa file gốc
                os.remove(input_path)
                # Đổi tên file tạm thành file gốc
                os.rename(temp_path, input_path)
                
                # Tính kích thước sau khi nén
                self.compressed_size_mb = os.path.getsize(input_path) / (1024**2)
                self.compression_ratio = (self.original_size_mb - self.compressed_size_mb) / self.original_size_mb * 100
                
                print(f"\n🎉 Nén thành công NGAY TẠI CHỖ!")
                print(f"📊 Kết quả:")
                print(f"   - File gốc: {self.original_size_mb:.2f} MB")
                print(f"   - File sau nén: {self.compressed_size_mb:.2f} MB")
                print(f"   - Tỷ lệ nén: {self.compression_ratio:.1f}%")
                print(f"   - Vị trí: {input_path} (đã được thay thế)")
                
                # Thông tin thêm về dữ liệu
                print(f"\n📈 Thông tin dữ liệu:")
                print(f"   - Số dòng: {len(df_optimized):,}")
                print(f"   - Số cột: {len(df_optimized.columns)}")
                
                return True
            else:
                print(f"\n⚠️  File tạm ({temp_size_mb:.2f} MB) LỚN HƠN file gốc ({self.original_size_mb:.2f} MB)")
                print("   Giữ nguyên file gốc và xóa file tạm...")
                os.remove(temp_path)
                return False
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file: {str(e)}")
            # Xóa file tạm nếu có lỗi
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
    
    def compress_csv_to_parquet_replace(self, input_path, backup_original=True):
        """
        Nén file CSV và thay thế bằng file Parquet cùng tên (đổi đuôi .csv -> .parquet)
        
        Args:
            input_path: Đường dẫn file CSV gốc
            backup_original: Có tạo backup file gốc không
            
        Returns:
            Đường dẫn file Parquet mới
        """
        print(f"\n📁 Đang xử lý file: {input_path}")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(input_path):
            print(f"❌ File không tồn tại: {input_path}")
            return None
        
        # Tạo đường dẫn file Parquet (đổi đuôi .csv -> .parquet)
        parquet_path = str(Path(input_path).with_suffix('.parquet'))
        
        # Tính kích thước file gốc
        self.original_size_mb = os.path.getsize(input_path) / (1024**2)
        print(f"📊 Kích thước CSV gốc: {self.original_size_mb:.2f} MB")
        
        try:
            # Tạo backup nếu cần
            if backup_original:
                backup_path = f"{input_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(input_path, backup_path)
                print(f"💾 Đã tạo backup CSV: {backup_path}")
            
            # Đọc file CSV với chunksize
            print("📖 Đang đọc file CSV...")
            chunksize = 100000
            chunks = []
            
            for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize)):
                chunks.append(chunk)
                print(f"   Đã đọc chunk {i+1}: {len(chunk):,} dòng", end='\r')
            
            print(f"\n✅ Đã đọc toàn bộ file: {sum(len(c) for c in chunks):,} dòng")
            
            # Kết hợp tất cả chunks
            df = pd.concat(chunks, ignore_index=True)
            
            # Tối ưu kiểu dữ liệu
            df_optimized = self.optimize_dtypes(df)
            
            # Lưu thành Parquet
            print("💾 Đang lưu file Parquet...")
            df_optimized.to_parquet(
                parquet_path,
                engine='pyarrow',
                compression='gzip',
                index=False
            )
            
            # Tính kích thước file Parquet
            self.compressed_size_mb = os.path.getsize(parquet_path) / (1024**2)
            self.compression_ratio = (self.original_size_mb - self.compressed_size_mb) / self.original_size_mb * 100
            
            # Xóa file CSV gốc (sau khi đã lưu Parquet thành công)
            os.remove(input_path)
            
            print(f"\n🎉 Đã thay thế CSV bằng Parquet!")
            print(f"📊 Kết quả:")
            print(f"   - CSV gốc: {self.original_size_mb:.2f} MB")
            print(f"   - Parquet mới: {self.compressed_size_mb:.2f} MB")
            print(f"   - Tỷ lệ nén: {self.compression_ratio:.1f}%")
            print(f"   - File CSV đã được xóa")
            print(f"   - File Parquet mới: {parquet_path}")
            
            return parquet_path
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý file: {str(e)}")
            return None
    
    def compress_multiple_files_in_place(self, file_paths, method='csv_compressed', backup_original=True):
        """
        Nén nhiều file CSV NGAY TẠI CHỖ
        
        Args:
            file_paths: Danh sách đường dẫn file
            method: 'csv_compressed' (giữ CSV) hoặc 'parquet_replace' (đổi sang Parquet)
            backup_original: Có tạo backup file gốc không
            
        Returns:
            Danh sách kết quả
        """
        print(f"🚀 Bắt đầu nén {len(file_paths)} file NGAY TẠI CHỖ...")
        print(f"📁 Phương pháp: {method}")
        print("="*50)
        
        results = []
        
        for i, file_path in enumerate(file_paths, 1):
            print(f"\n📦 File {i}/{len(file_paths)}: {Path(file_path).name}")
            
            if method == 'csv_compressed':
                success = self.compress_csv_in_place(file_path, backup_original)
                if success:
                    results.append({
                        'file': file_path,
                        'status': 'success',
                        'original_size_mb': self.original_size_mb,
                        'compressed_size_mb': self.compressed_size_mb,
                        'compression_ratio': self.compression_ratio,
                        'method': 'CSV compressed (gzip)'
                    })
                else:
                    results.append({
                        'file': file_path,
                        'status': 'failed',
                        'method': 'CSV compressed (gzip)'
                    })
            
            elif method == 'parquet_replace':
                parquet_path = self.compress_csv_to_parquet_replace(file_path, backup_original)
                if parquet_path:
                    results.append({
                        'file': file_path,
                        'new_file': parquet_path,
                        'status': 'success',
                        'original_size_mb': self.original_size_mb,
                        'compressed_size_mb': self.compressed_size_mb,
                        'compression_ratio': self.compression_ratio,
                        'method': 'Replaced with Parquet'
                    })
                else:
                    results.append({
                        'file': file_path,
                        'status': 'failed',
                        'method': 'Replaced with Parquet'
                    })
            
            print("="*50)
        
        # Tổng kết
        if results:
            print("\n📊 TỔNG KẾT NÉN NGAY TẠI CHỖ")
            print("="*50)
            
            successful = [r for r in results if r['status'] == 'success']
            failed = [r for r in results if r['status'] == 'failed']
            
            print(f"📁 Tổng số file: {len(results)}")
            print(f"✅ Thành công: {len(successful)}")
            print(f"❌ Thất bại: {len(failed)}")
            
            if successful:
                total_original = sum(r['original_size_mb'] for r in successful)
                total_compressed = sum(r['compressed_size_mb'] for r in successful)
                avg_ratio = sum(r['compression_ratio'] for r in successful) / len(successful)
                
                print(f"\n📊 Tổng kích thước gốc: {total_original:.2f} MB")
                print(f"📊 Tổng kích thước sau nén: {total_compressed:.2f} MB")
                print(f"🎯 Tiết kiệm tổng: {total_original - total_compressed:.2f} MB")
                print(f"📈 Tỷ lệ nén trung bình: {avg_ratio:.1f}%")
            
            # Lưu báo cáo
            report_df = pd.DataFrame(results)
            report_path = 'compression_in_place_report.csv'
            report_df.to_csv(report_path, index=False)
            print(f"\n📄 Báo cáo đã được lưu: {report_path}")
        
        return results

def get_system_info():
    """Lấy thông tin hệ thống"""
    print("💻 THÔNG TIN HỆ THỐNG")
    print("="*50)
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print("="*50)

def main():
    """Hàm chính"""
    print("="*50)
    print("     CSV FILE COMPRESSOR - IN-PLACE")
    print("     Nén CSV NGAY TẠI CHỖ (giảm kích thước file)")
    print("="*50)
    
    # Hiển thị thông tin hệ thống
    get_system_info()
    
    # Tạo compressor
    compressor = CSVCompressor(target_size_mb=25)
    
    # DANH SÁCH FILE CẦN NÉN - THAY THẾ NGAY TẠI CHỖ
    files_input = ('/app/data/application_train.csv', '/app/data/df_processed.csv')
    file_paths = list(files_input)
    
    # Kiểm tra file tồn tại
    valid_files = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            valid_files.append(file_path)
            size_mb = os.path.getsize(file_path) / (1024**2)
            print(f"✅ Tìm thấy: {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ Không tìm thấy: {file_path}")
    
    if not valid_files:
        print("❌ Không có file nào để nén!")
        exit(1)
    
    print(f"\n📁 Sẽ nén NGAY TẠI CHỖ {len(valid_files)} file:")
    for f in valid_files:
        size_mb = os.path.getsize(f) / (1024**2)
        print(f"   • {Path(f).name} ({size_mb:.1f} MB)")
    
    # Hỏi phương pháp nén
    print("\n📂 PHƯƠNG PHÁP NÉN:")
    print("1. Giữ nguyên định dạng CSV (nén gzip) - vẫn là file .csv")
    print("2. Đổi sang Parquet (.csv → .parquet) - nén tốt hơn")
    print("3. Tạo file mới, giữ nguyên file gốc")
    
    method_choice = input("\n👉 Chọn phương pháp (1-3): ").strip()
    
    # Thực hiện nén
    if method_choice == '1':
        print("\n🚀 Đang nén CSV NGAY TẠI CHỖ (giữ định dạng CSV)...")
        results = compressor.compress_multiple_files_in_place(
            valid_files, 
            method='csv_compressed',
            backup_original=True
        )
    elif method_choice == '2':
        print("\n🚀 Đang thay thế CSV bằng Parquet...")
        results = compressor.compress_multiple_files_in_place(
            valid_files,
            method='parquet_replace',
            backup_original=True
        )
    elif method_choice == '3':
        print("\n🚀 Đang nén và tạo file mới...")
        # Gọi phương thức cũ để tạo file mới
        results = compressor.compress_multiple_files(valid_files, 'parquet')
    else:
        print("⚠️  Lựa chọn không hợp lệ, mặc định nén CSV tại chỗ")
        results = compressor.compress_multiple_files_in_place(
            valid_files,
            method='csv_compressed',
            backup_original=True
        )
    
    # Kiểm tra kết quả cuối cùng
    print("\n🔍 KIỂM TRA KẾT QUẢ:")
    print("="*50)
    
    for file_path in valid_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024**2)
            print(f"✅ {Path(file_path).name}: {size_mb:.2f} MB")
        else:
            # Có thể file đã được đổi thành .parquet
            parquet_path = str(Path(file_path).with_suffix('.parquet'))
            if os.path.exists(parquet_path):
                size_mb = os.path.getsize(parquet_path) / (1024**2)
                print(f"✅ {Path(parquet_path).name}: {size_mb:.2f} MB (đã đổi từ CSV)")
            else:
                print(f"❌ File không tồn tại: {file_path}")
    
    print("\n✨ Hoàn tất!")

if __name__ == "__main__":
    main()