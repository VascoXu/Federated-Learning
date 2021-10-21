from __future__ import print_function

import logging

import grpc
import federated_pb2
import federated_pb2_grpc


class Client:
    def __init__(self):
        pass

    def start_client(self, server_address):
        with grpc.insecure_channel(server_address) as channel:
            stub = federated_pb2_grpc.FederatedStub(channel)
            modelString = federated_pb2.Model()
            response = stub.Join(modelString)
        print("Client received: " + response.message)

        # try:
        #     logging.basicConfig()
        #     run()
        # finally:
        #     pass