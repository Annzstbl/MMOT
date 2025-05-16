from setuptools import setup, find_packages

setup(
    name="hsmot",              # 包的名字
    version="0.2.0",                  # 版本号
    author="LTH",               # 作者信息
    author_email="lth@163.com",  # 作者邮箱
    description="A short description",     # 简要描述
    # long_description=open("README.md").read(),  # 可选，详细描述
    long_description_content_type="text/markdown",
    # url="https://github.com/your_repo",   # 项目地址（如 GitHub 链接）
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
