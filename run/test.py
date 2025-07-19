from run.val import validate  # 复用验证逻辑

"""
    测试
"""
def test_model(model, loader, device):
    # 混淆矩阵：分析各类别识别情况
    metrics = validate(model, loader, device)
    # 可添加测试特有的处理
    metrics['description'] = "Final test results"
    return metrics