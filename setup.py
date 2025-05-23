import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'field_camera_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'models/camera'), glob('models/camera/*')),
        (os.path.join('share', package_name, 'models/man'), glob('models/man/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='james',
    maintainer_email='james@todo.todo',
    description='TODO: Package description',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'object_detection = field_camera_sim.object_detection:main'
        ],
    },
)
