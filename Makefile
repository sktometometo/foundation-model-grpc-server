all:
	python3.8 -m grpc_tools.protoc -I./proto --python_out=./src/LAVIS_grpc_server --pyi_out=./src/LAVIS_grpc_server --grpc_python_out=./src/LAVIS_grpc_server ./proto/lavis-server.proto

clean:
	rm ./src/LAVIS_grpc_server/*_pb2.py ./src/LAVIS_grpc_server/*_pb2.pyi ./src/LAVIS_grpc_server/*_pb2_grpc.py
