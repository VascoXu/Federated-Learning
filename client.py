from argparse import ArgumentParser

import grpc_client

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--server_address",
        type=str,
        default="localhost:50051",
        help="gRPC server address (default)"
    )
    
    args = parser.parse_args()

    client = grpc_client.Client()
    client.start_client(args.server_address)
