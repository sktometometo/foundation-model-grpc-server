import os

from setuptools import setup

try:
    from catkin_pkg.python_setup import generate_distutils_setup

    ros_enabled = True
except ModuleNotFoundError:
    ros_enabled = False


def get_requirements():
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    requirement_path = lib_folder + (
        "/requirements_ros.txt" if ros_enabled else "/requirements.txt"
    )
    install_requires = []
    if os.path.isfile(requirement_path):
        with open(requirement_path, encoding="utf-8") as f:
            install_requires = f.read().splitlines()
    return install_requires


package_info = {
    "packages": [
        "foundation_model_grpc_core",
        "foundation_model_grpc_interface",
        "foundation_model_grpc_utils",
    ],
    "package_dir": {"": "src"},
    "install_requires": get_requirements(),
    "entry_points": {
        "console_scripts": [
            "dummy_command = foundation_model_grpc_core:dummy_command",
            "download_lavis_model = foundation_model_grpc_core.lavis_server:download_model",
            "download_llama_adapter_server_model = foundation_model_grpc_core.llama_adapter_server:download_model",
            "run_llama_adapter_server = foundation_model_grpc_core.llama_adapter_server:run_server",
        ],
        "gui_scripts": [
            "run_lavis_server = foundation_model_grpc_core.lavis_server:main_server",
            "sample_client = foundation_model_grpc_core:main_client_sample",
            "sample_continuous_captioning = foundation_model_grpc_core.continuous_captioning:main",
        ],
    },
    "classifiers": [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
}

if ros_enabled:
    d = generate_distutils_setup(**package_info)
else:
    d = package_info
    d.update(
        {
            "name": "foundation_model_grpc_server",
            "description": "This is a package provides grpc server for foundation model.",
            "author": "Koki Shinjo",
            "author_email": "sktometometo@gmail.com",
            "url": "https://github.com/sktometometo/foundation-model-grpc-server",
            "version": "0.2.0",
        }
    )

setup(**d)
