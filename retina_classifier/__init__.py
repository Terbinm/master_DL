"""
Retina Damage Classification System
視網膜損傷分類系統

基於 EfficientNet-B0 的視網膜 OCT 圖像分類系統
針對 RTX 4090 + PyTorch 2.5.1 + CUDA 12.1 環境優化
"""

__version__ = "1.0.0"
__title__ = "Retina Damage Classification"
__description__ = "視網膜損傷分類系統 - 基於深度學習的 OCT 圖像四類別分類"
__author__ = "Retina Classification Team"
__license__ = "Academic Use Only"
__copyright__ = "Copyright 2025"

# 版本信息
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

# 系統需求
SYSTEM_REQUIREMENTS = {
    'python': '>=3.8',
    'pytorch': '>=2.0.0',
    'cuda': '>=11.8',
    'gpu_memory': '>=8GB',
    'ram': '>=16GB'
}

# 支援的分類類別
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# 模組導入
try:
    from .config import Config
    from .models import RetinaClassifier, create_retina_model
    from .utils import RetinaTrainer, create_data_loaders, create_tensorboard_visualizer

    __all__ = [
        # Core classes
        'Config',
        'RetinaClassifier',
        'RetinaTrainer',

        # Factory functions
        'create_retina_model',
        'create_data_loaders',
        'create_tensorboard_visualizer',

        # Constants
        'CLASS_NAMES',
        'VERSION_INFO',
        'SYSTEM_REQUIREMENTS'
    ]

except ImportError as e:
    # 如果在開發環境中，可能會有導入問題
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = [
        'CLASS_NAMES',
        'VERSION_INFO',
        'SYSTEM_REQUIREMENTS'
    ]

def get_version():
    """獲取版本信息"""
    return __version__

def get_system_info():
    """獲取系統信息"""
    import torch
    import platform

    info = {
        'system': platform.system(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__ if torch else 'Not installed',
        'cuda_available': torch.cuda.is_available() if torch else False,
        'package_version': __version__
    }

    if torch and torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0)
        })

    return info

def print_system_info():
    """打印系統信息"""
    print(f"{__title__} v{__version__}")
    print("=" * 50)

    info = get_system_info()
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("=" * 50)

# 版權信息
def show_license():
    """顯示授權信息"""
    license_text = f"""
{__title__} v{__version__}
{__description__}

{__copyright__}
License: {__license__}

This software is provided for academic and research purposes only.
Commercial use is prohibited without explicit permission.

Supported Classifications:
{', '.join(CLASS_NAMES)}

For more information, please refer to the README.md file.
    """
    print(license_text.strip())