from .models import (
    Document,
    Chunk,
)


def chunk_document_fixed( document: Document, chunk_size: int,) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks: list[Chunk] = []

    # enumerate()给每个位置编号。遍历时，`enumerate()` 每次给start提供一对值；
    # range举例 range(0, 10, 4)意思是：从 0 开始，每次加 4，到 10 之前停止。依次得到：0、4、8
    for chunk_index, start in enumerate(range(0, len(document.text), chunk_size)):
        # 截取当前段，包含 start，不包含结束位置。
        chunk_text = document.text[start : start + chunk_size]

        # 为当前段创建数据对象，并记录来源文档。
        chunk = Chunk(
            id=f"{document.id}-chunk-{chunk_index}",
            document_id=document.id,
            text=chunk_text,
        )

        # 将当前段加入结果列表。
        chunks.append(chunk)

    # 所有段处理完后，返回完整列表。
    return chunks


def chunk_document_by_paragraph(document: Document,) -> list[Chunk]:
    """按连续两个换行分段，清理首尾空白并跳过空段。"""

    # 收集最终生成的 Chunk。
    chunks: list[Chunk] = []

    # 按段落分隔符切开，得到字符串列表。“连续两个换行”写成 \n\n，表示两段文字之间有一条空行。
    paragraphs = document.text.split("\n\n")

    for paragraph in paragraphs:
        # 删除当前段落首尾的空白。
        paragraph = paragraph.strip()

        # 清理后如果没有内容，直接处理下一段。
        if not paragraph:
            continue

        # 已生成的段数，就是新段的序号。
        chunk_index = len(chunks)

        # 创建当前段，并记录它来自哪篇文档。f"" 是格式化字符串：可以把变量的值放进文字里。例如 text = f"你好，{name}"
        chunk = Chunk(
            id=f"{document.id}-chunk-{chunk_index}",
            document_id=document.id,
            text=paragraph,
        )

        # 将当前段加入结果列表。
        chunks.append(chunk)

    # 全部段落处理结束后，返回结果。
    return chunks