from setuptools import find_packages, setup

setup(
    name='3d_printing_quality_prediction',
    version='0.1.0',
    description='使用機器學習預測3D列印品質指標',
    author='Author',
    author_email='author@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'joblib>=1.0.0',
    ],
    python_requires='>=3.12',
)