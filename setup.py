import os
from distutils.core import setup


def get_requirements():
  lib_folder = os.path.dirname(os.path.realpath(__file__))
  requirement_path = lib_folder + '/requirements.txt'
  install_requires = [
  ]  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
  if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
      install_requires = f.read().splitlines()
  return install_requires


package_info = {
    "packages": ['LAVIS_grpc_server'],
    "package_dir": {
        '': 'src'
    },
    "install_requires":
        get_requirements(),
    "entry_points": {
        'console_scripts': [
            "download_model_cache = LAVIS_grpc_server:download_model_cache",
        ],
        'gui_scripts': [
            "run_lavis_server = LAVIS_grpc_server:main_server",
            "sample_lavis_client = LAVIS_grpc_server:main_client_sample",
            "sample_continuous_captioning = LAVIS_grpc_server.continuous_captioning:main"
        ]
    },
    "classifiers": [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
}

try:
  from catkin_pkg.python_setup import generate_distutils_setup

  d = generate_distutils_setup(**package_info)
except ModuleNotFoundError as e:
  d = package_info
  d.update({
      'name': 'LAVIS_grpc_server',
      'version': '0.0.1',
  })

setup(**d)
