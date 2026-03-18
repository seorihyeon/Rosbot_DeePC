from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'rosbot_local_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Local bringup package for ROSbot simulation with ground truth bridge',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim_ready_notifier = rosbot_local_bringup.sim_ready_notifier:main',
            'reset_rosbot_server = rosbot_local_bringup.reset_server:main',
        ],
    },
)
