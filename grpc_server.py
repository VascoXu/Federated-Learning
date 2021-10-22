from concurrent import futures
from io import BytesIO
from typing import cast

import numpy as np

import grpc
import federated_pb2
import federated_pb2_grpc


def bytes_to_ndarray(tensor: bytes) -> np.ndarray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)


class FederatedServicer(federated_pb2_grpc.FederatedServicer):
    def GetServerResponse(self, request_iterator, context):
        for message in request_iterator:
            yield message

    def Join(self, request, context):
        """Join model"""
        
        # print(request.weights)
        weights = [bytes_to_ndarray(ndarray) for ndarray in request.weights]
        print(weights)
        response = federated_pb2.Empty()
        response.message = "testing"
        return response


class Server:
    def __init__(self):
        pass

    def start_server(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        federated_pb2_grpc.add_FederatedServicer_to_server(FederatedServicer(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()

        # logging.basicConfig()
        # serve()