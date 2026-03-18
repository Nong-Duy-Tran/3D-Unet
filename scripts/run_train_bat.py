import subprocess
import os

def run_batch_file(bat_path):
    # Kiểm tra xem file .bat có tồn tại không
    if not os.path.exists(bat_path):
        print(f"Lỗi: Không tìm thấy file tại {bat_path}")
        return

    print(f"--- Đang bắt đầu thực thi: {bat_path} ---")
    
    try:
        # Chạy file .bat và hiển thị output trực tiếp ra console
        # shell=True là bắt buộc để chạy file .bat trên Windows
        result = subprocess.run([bat_path], shell=True, check=True)
        
        if result.returncode == 0:
            print("--- Hoàn thành tất cả các Fold thành công! ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Có lỗi xảy ra trong quá trình chạy file .bat: {e} ---")
    except Exception as e:
        print(f"--- Lỗi không xác định: {e} ---")

if __name__ == "__main__":
    # Thay đổi đường dẫn này cho đúng với máy của bạn
    path_to_bat = r"D:\YEAR 3\Lab\3D-Unet\scripts\train_cv5_fold.bat"
    run_batch_file(path_to_bat)