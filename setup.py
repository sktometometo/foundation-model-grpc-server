import os
from distutils.core import setup

try:
  from catkin_pkg.python_setup import generate_distutils_setup
  ros_enabled = True
except:
  ros_enabled = False


def get_requirements():
  lib_folder = os.path.dirname(os.path.realpath(__file__))
  requirement_path = lib_folder + ('/requirements_ros.txt'
                                   if ros_enabled else '/requirements.txt')
  install_requires = []
  if os.path.isfile(requirement_path):
    with open(requirement_path, encoding='utf-8') as f:
      install_requires = f.read().splitlines()
  return install_requires


package_info = {
    "packages": [
        'foundation_model_grpc_core', 'foundation_model_grpc_interface',
        'foundation_model_grpc_utils'
    ],
    "package_dir": {
        '': 'src'
    },
    "install_requires":
        get_requirements(),
    "entry_points": {
        'console_scripts': [
            "download_model_cache = foundation_model_grpc_core.lavis_server:download_model_cache",
            "run_llama_adapter_server = foundation_model_grpc_core.llama_adapter_server:run_server",
        ],
        'gui_scripts': [
            "run_lavis_server = foundation_model_grpc_core.lavis_server:main_server",
            "sample_client = foundation_model_grpc_core:main_client_sample",
            "sample_continuous_captioning = foundation_model_grpc_core.continuous_captioning:main"
        ]
    },
    "classifiers": [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
}

if ros_enabled:
  d = generate_distutils_setup(**package_info)
else:
  d = package_info
  d.update({
      'name': 'foundation_model_grpc_server',
      'version': '0.0.1',
  })

setup(**d)
