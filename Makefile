all:
	python3.8 -m grpc_tools.protoc -I./proto --python_out=./src/foundation_model_grpc_interface --pyi_out=./src/foundation_model_grpc_interface --grpc_python_out=./src/foundation_model_grpc_interface ./proto/lavis-server.proto
	sed -i "s/import lavis_server_pb2 as lavis__server__pb2/from foundation_model_grpc_interface import lavis_server_pb2 as lavis__server__pb2/" ./src/foundation_model_grpc_interface/lavis_server_pb2_grpc.py

clean:
	rm ./src/foundation_model_grpc_interface/*_pb2.py ./src/foundation_model_grpc_interface/*_pb2.pyi ./src/foundation_model_grpc_interface/*_pb2_grpc.py
