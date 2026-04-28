from setuptools import setup
from glob import glob
import os

package_name = 'rosbot_deepc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'cvxpy',
    ],
    zip_safe=True,
    maintainer='skim',
    maintainer_email='seu7704@dgist.ac.kr',
    description='DeePC package for Rosbot',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'reference_collect_node = rosbot_deepc.reference_collect_node:main',
        'random_collect_node = rosbot_deepc.random_collect_node:main',
        'prbs_collect_node = rosbot_deepc.prbs_collect_node:main',
        'deepc_node = rosbot_deepc.deepc_node:main',
        ],
    },
)
