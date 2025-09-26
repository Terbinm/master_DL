import os
import sys


def print_tree(output_file, exclude_dirs, exclude_files):
    def generate_tree(startpath):
        tree_str = ""
        for root, dirs, files in os.walk(startpath):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            level = root.replace(startpath, '').count(os.sep)
            indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
            tree_str += f'{indent}{os.path.basename(root)}/\n'

            subindent = '│   ' * level + '├── '
            for f in files:
                if f not in exclude_files:
                    tree_str += f'{subindent}{f}\n'
        return tree_str

    return generate_tree('.')


def serialize_files(output_file=""):
    # 確認當前執行的檔案名稱
    current_script = os.path.basename(sys.argv[0])

    # 預設的排除目錄和內容
    exclude_dirs = {'.venv', '.idea', '.git', '__pycache__', '.pytest_cache', 'temp', 'scripts'
                    , 'data', 'logs', 'results', 'checkpoints'}
    exclude_content = {'.md', ".db", ".rds", ".pptx", ".zip", ".joblib", ".png", ".jpg", ".jpeg"}

    # 設定的檔案名稱 - 如果腳本名稱不是 'output_LLM.py' 則終止執行
    expected_script_name = 'output_LLM.py'
    if current_script != expected_script_name:
        print(f"錯誤: 此腳本必須以 '{expected_script_name}' 的名稱執行，目前檔名為 '{current_script}'")
        sys.exit(1)

    # 排除當前執行的腳本
    exclude_files = {current_script}

    # 行數限制設定：檔案類型及其最大行數
    line_limits = {
        '.csv': 10,  # CSV檔案超過10行後省略
        '.json': 20,  # JSON檔案超過20行後省略
        '.log': 30,  # LOG檔案超過30行後省略
        # 可以根據需要添加更多限制
    }

    # 用於存儲檢測到亂碼的檔案列表
    garbled_files = []

    if not output_file:
        # output_file = "all_files.md"
        root_dir = os.path.basename(os.path.abspath('.'))
        output_file = f"{root_dir}---all_files.md"

    with open(output_file, 'w', encoding='utf-8') as f:
        # 首先寫入目錄樹結構
        f.write("# ")
        f.write(output_file)
        f.write("\n")
        f.write("# 目錄結構\n```\n")
        f.write(print_tree(output_file, exclude_dirs, exclude_files))
        f.write("```\n---\n")

        # 接著寫入檔案內容
        for root, dirs, files in os.walk('.', topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file == output_file or file in exclude_files:
                    continue

                file_path = os.path.join(root, file)[2:]
                file_ext = os.path.splitext(file)[1]

                f.write(f"\n{file_path}:\n")
                if file_ext not in exclude_content:
                    try:
                        # 嘗試不同的編碼方式
                        encodings = ['utf-8', 'utf-16', 'big5', 'cp950', 'gb18030']
                        content = None
                        has_garbled = False

                        for encoding in encodings:
                            try:
                                with open(os.path.join(root, file), 'r', encoding=encoding) as content_file:
                                    content = content_file.read()

                                    # 簡單的亂碼檢測：檢查是否包含替換字符
                                    if ' ' in content:
                                        has_garbled = True
                                        garbled_files.append(file_path)
                                break
                            except UnicodeDecodeError:
                                continue

                        if content is None:
                            # 如果所有編碼都失敗，使用二進制模式讀取
                            with open(os.path.join(root, file), 'rb') as content_file:
                                content = content_file.read().decode('utf-8', errors='replace')

                                # 檢查替換字符
                                if ' ' in content:
                                    has_garbled = True
                                    garbled_files.append(file_path)

                        # 檢查是否需要限制行數
                        if file_ext in line_limits:
                            lines = content.splitlines()
                            max_lines = line_limits[file_ext]

                            if len(lines) > max_lines:
                                f.write("```\n")
                                # 只輸出限制的行數
                                f.write('\n'.join(lines[:max_lines]))
                                f.write(f"\n\n... (省略剩餘 {len(lines) - max_lines} 行內容) ...\n")
                                f.write("```\n")
                            else:
                                f.write("```\n")
                                f.write(content)
                                f.write("\n```\n")
                        else:
                            f.write("```\n")
                            f.write(content)
                            f.write("\n```\n")

                        # 亂碼警告
                        if has_garbled:
                            f.write(f"\n**注意：此檔案可能包含亂碼，請考慮添加至排除列表。**\n")

                    except Exception as e:
                        print(f"無法讀取檔案 {file_path}: {str(e)}")
                        garbled_files.append(file_path)

                else:
                    f.write("```\n[內容已忽略]\n```\n")
                f.write("---\n")

        # 將亂碼檔案列表寫入文件末尾
        if garbled_files:
            f.write("\n\n# 檢測到亂碼的檔案列表\n")
            f.write("以下檔案可能包含亂碼，建議將它們添加到排除列表中：\n\n")
            for garbled_file in garbled_files:
                f.write(f"- {garbled_file}\n")

    # 在控制台輸出亂碼檔案列表
    if garbled_files:
        print("\n檢測到以下檔案可能包含亂碼：")
        for garbled_file in garbled_files:
            print(f"- {garbled_file}")
        print("\n建議將這些檔案添加至排除列表。")


if __name__ == "__main__":
    serialize_files()