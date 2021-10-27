from concurrent import futures

import numpy as np

import grpc
import federated_pb2
import federated_pb2_grpc

from helpers import bytes_to_ndarray


# Keep track of global weights
global_weights = []

# Keep track of participants
participants = []


class FederatedServicer(federated_pb2_grpc.FederatedServicer):
    def Join(self, request, context):
        """Join model"""

        # Deserialize the wweights        
        weights = [bytes_to_ndarray(ndarray) for ndarray in request.weights]
        response = federated_pb2.Empty()
        response.message = "Success!"
        return response

    
    def Register(self, request, context):
        """Register device"""
        participants.append(request)


class Server:
    def __init__(self):
        pass

    def start_server(self):
        """Start server using gRPC"""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        federated_pb2_grpc.add_FederatedServicer_to_server(FederatedServicer(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        server.wait_for_termination()