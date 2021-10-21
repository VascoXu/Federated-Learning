from concurrent import futures
import logging

import grpc
import federated_pb2
import federated_pb2_grpc


class FederatedServicer(federated_pb2_grpc.FederatedServicer):
    def GetServerResponse(self, request_iterator, context):
        for message in request_iterator:
            yield message

    def Join(self, request, context):
        """Join model"""
        
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