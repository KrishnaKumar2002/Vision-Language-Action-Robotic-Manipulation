from setuptools import setup

package_name = 'vla_task_planner'

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
    description='Natural language parsing and task decomposition for VLA manipulation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'language_task_node = vla_task_planner.language_task_node:main',
        ],
    },
)
