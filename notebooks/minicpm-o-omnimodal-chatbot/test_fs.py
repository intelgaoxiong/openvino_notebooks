import os

def test_file_permissions(file_path):
    try:
        # 尝试创建并写入文件
        with open(file_path, 'w') as f:
            f.write("Testing write permissions.\n")
        print(f"Successfully wrote to {file_path}")

        # 尝试读取文件
        with open(file_path, 'r') as f:
            content = f.read()
        print(f"Successfully read from {file_path}: {content}")

        # 尝试删除文件
        os.remove(file_path)
        print(f"Successfully deleted {file_path}")

    except PermissionError:
        print(f"Permission denied for {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 测试路径
test_file_path = os.path.join(os.path.expanduser("~"), "Desktop", "test_permissions.txt")
test_file_permissions(test_file_path)