import grpc_server

if __name__ == "__main__":
    server = grpc_server.Server()
    server.start_server()