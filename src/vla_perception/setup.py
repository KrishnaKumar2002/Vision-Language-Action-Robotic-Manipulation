from setuptools import setup

package_name = 'vla_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Krishna Kumar',
    maintainer_email='krishna@example.com',
    description='Simulated 3D vision and object detection for VLA robot manipulation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'detector_node = vla_perception.detector_node:main',
        ],
    },
)
