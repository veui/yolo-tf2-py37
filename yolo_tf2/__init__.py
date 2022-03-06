# import sys
#
# __author__ = 'alternativebug'
# __email__ = 'alternativebug'
# __license__ = 'MIT'
# __version__ = 1.5
#
# if sys.version_info < (3, 8):
#     print(f'Yolo-tf2 {__version__} requires Python 3.9+')
#     sys.exit(1)
#
# dependencies = (
#     'pandas',
#     'lxml',
#     'imagesize',
#     'seaborn',
#     'tensorflow',
#     'numpy',
#     'matplotlib',
#     'imgaug',
#     'imagecorruptions',
#     'tensorflow_addons',
#     'configparser',
#     'cv2',
#     'PyQt5',
# )
# missing_dependencies = []
#
# for dependency in dependencies:
#     try:
#         __import__(dependency)
#     except ImportError as e:
#         missing_dependencies.append(f"{dependency}: {e}")
#
# if missing_dependencies:
#     dependencies = '\n'.join(missing_dependencies)
#     raise ImportError(f'Unable to import required dependencies:\n{dependencies}')
#
# del dependencies, dependency, missing_dependencies, sys
