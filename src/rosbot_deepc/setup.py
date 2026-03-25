from setuptools import find_packages, setup
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
    install_requires=['setuptools'],
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
             'collect_node = rosbot_deepc.collect_node:main',
        ],
    },
)
