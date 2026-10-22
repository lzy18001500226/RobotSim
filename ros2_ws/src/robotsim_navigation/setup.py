from setuptools import setup

package_name = "robotsim_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RobotSim Maintainers",
    maintainer_email="maintainers@example.com",
    description="Issue 80 research adapter for measured-odometry point-goal navigation.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "point_goal_adapter = robotsim_navigation.point_goal_adapter:main",
        ],
    },
)