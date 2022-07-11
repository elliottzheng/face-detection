from os import path as os_path

from setuptools import setup

this_directory = os_path.abspath(os_path.dirname(__file__))


# 读取文件内容
def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding="utf-8") as f:
        long_description = f.read()
    return long_description


# 获取依赖
def read_requirements(filename):
    return [
        line.strip()
        for line in read_file(filename).splitlines()
        if not line.startswith("#")
    ]


setup(
    name="face-detection",
    version='1.0.5',
    description="Fast and reliable face detection with RetinaFace",
    author="Elliott Zheng",
    author_email="admin@hypercube.top",
    url="https://github.com/elliottzheng/RetinaFace",
    license="MIT",
    keywords="face-detection pytorch RetinaFace",
    project_urls={
        "Documentation": "https://github.com/elliottzheng/face-detection",
        "Source": "https://github.com/elliottzheng/face-detection",
        "Tracker": "https://github.com/elliottzheng/face-detection/issues",
    },
    long_description=read_file("README.md"),  # 读取的Readme文档内容
    long_description_content_type="text/markdown",  # 指定包文档格式为markdown
    packages=["face_detection"],
    install_requires=["numpy", "torch", "torchvision"],
    package_data={'face_detection': ['weights/*.pth']}
)
