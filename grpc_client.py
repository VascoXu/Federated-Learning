from __future__ import print_function

import logging

import grpc
import federated_pb2
import federated_pb2_grpc


def make_message(message):
    return federated_pb2.Empty(
        message=message
    )


def generate_messages():
    messages = [
        make_message("One"),
        make_message("Two"),
        make_message("Three"),
        make_message("Four"),
        make_message("Five")
    ]
    for msg in messages:
        print(f"Sending server message: {msg.message}")
        yield msg


def send_message(stub):
    responses = stub.GetServerResponse(generate_messages())
    for response in responses:
        print(f"Message received from server: {response.message}")

class Client:
    def __init__(self):
        pass

    def start_client(self, server_address):
        with grpc.insecure_channel(server_address) as channel:
            stub = federated_pb2_grpc.FederatedStub(channel)
            send_message(stub)
            modelString = federated_pb2.Model()
            response = stub.Join(modelString)
        # print("Client received: " + response.message)

        # try:
        #     logging.basicConfig()
        #     run()
        # finally:
        #     pass