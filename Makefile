all:
	python -m grpc_tools.protoc -I./src --python_out=./src --pyi_out=./src --grpc_python_out=./src ./src/lavis-server.proto

clean:
	rm ./src/*_pb2.py ./src/*_pb2.pyi ./src/*_pb2_grpc.py
