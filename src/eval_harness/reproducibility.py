import hashlib


def compute_file_sha256(path: str) -> str:
    # 创建 SHA-256 计算对象
    hasher = hashlib.sha256()

    # 以二进制模式打开文件; "rb",r：读取; b：二进制模式。 file.read() 读取文件的全部字节
    with open(path, "rb") as file:
        content = file.read()

    # 把这些字节加入当前 SHA-256 计算
    hasher.update(content)  

    # 获取当前计算结果，并表示为十六进制字符串。
    return hasher.hexdigest()