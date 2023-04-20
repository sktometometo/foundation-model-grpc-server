cmake_minimum_required(VERSION 2.8.3)
project(LAVIS_grpc_server)

find_package(catkin REQUIRED COMPONENTS
  catkin_virtualenv
)

catkin_python_setup()

catkin_generate_virtualenv(
    PYTHON_INTERPRETER python3.8
    CHECK_VENV FALSE
)

catkin_package(
)

catkin_install_python(PROGRAMS scripts/sample_lavis_client.py scripts/download_model_cache.py scripts/run_lavis_server.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
