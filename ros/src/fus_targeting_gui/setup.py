from setuptools import find_packages, setup
from glob import glob

package_name = "fus_targeting_gui"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="tyler zeng",
    maintainer_email="tonizengedu@gmail.com",
    description=(
        "Skull-mesh point-picking GUI for tFUS acoustic characterization. "
        "Loads a mesh, click to target, plans and executes via MoveIt. "
        "See package.xml for the full description."
    ),
    license="BSD",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "targeting_gui = fus_targeting_gui.node_entry:main",
        ],
    },
)
