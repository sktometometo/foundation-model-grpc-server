cmake_minimum_required(VERSION 2.8.3)
project(foundation_model_grpc_server)

find_package(catkin REQUIRED COMPONENTS
  catkin_virtualenv
  message_generation
  sensor_msgs
  rospy
)

catkin_python_setup()

catkin_generate_virtualenv(
    PYTHON_INTERPRETER python3
    USE_SYSTEM_PACKAGES FALSE
    CHECK_VENV FALSE
)

add_service_files(
  FILES
  ImageCaptioning.srv
  VisualQuestionAnswering.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

catkin_install_python(PROGRAMS
    scripts/rosbridge_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES requirements.txt requirements_ros.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
