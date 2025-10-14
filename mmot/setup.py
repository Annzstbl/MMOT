from setuptools import setup, find_packages

setup(
    name="mmot",              # 包的名字
    version="1.0.0",                  # 版本号
    author="LTH",               # 作者信息
    author_email="tianhaoli1996@gmail.com",  # 作者邮箱
    description="Core library for MMOT",     # 简要描述
    long_description_content_type="text/markdown",
    packages=find_packages(),            # 自动发现项目中的所有包
    # install_requires=[
    #     # 依赖列表，例如：
    #     "numpy>=1.19.0",
    #     "torch>=1.8.0"
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
