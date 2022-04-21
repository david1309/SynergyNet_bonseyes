from setuptools import setup, find_packages

reqs = [
    "numpy==1.21.2",
    "pandas==1.4.1",
    "matplotlib==3.5.1",
    "scipy==1.8.0",
    "opencv-python==4.5.5.64",
    "tensorboard==2.8.0",

    "python-dateutil==2.8.2",
    "ijson==3.1.4",
    "tqdm == 4.60.0",
]

flame_reqs = [
    "chumpy",
    "pyrender==0.1.39",
    "trimesh==3.6.18",
    "smplx",
]

dev_reqs = [
    "black",
    "pylint",
    "pytest",
]

setup(
    name="synergynet",
    version="0.0.1",
    description="Model for performing facial landmarks and head pose detection",
    url="https://github.com/david1309/SynergyNet_bonseyes",

    author="David Alvarez, Damiano Binaghi, Artificialy SA",
    author_email="artificialy@artificialy.com",

    license="restrictive, all right reserved by Artificialy SA",
    keywords=["facial landmarks", "3DMM", "morpable model"],
    include_package_data=True,
    packages=find_packages(),

    install_requires=reqs,
    extras_require={
        'dev': dev_reqs + reqs,
        'flame': flame_reqs + reqs,
    }
)
