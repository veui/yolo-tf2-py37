from setuptools import find_packages, setup

install_requires = open('requirements.txt').read().splitlines()

setup(
    name='yolo_tf2',
    version='1.5',
    packages=find_packages(),
    url='https://github.com/alternativebug/yolo-tf2',
    license='MIT',
    author='schismantics',
    author_email='alternativebug@outlook.com',
    description='yolo(v3/v4) implementation in keras and tensorflow 2.7',
    install_requires=install_requires,
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'yolotf2=yolo_tf2.cli:execute',
        ],
    },
)
