from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.SessionManager import SessionManager
from kortex_api.RouterClient import RouterClient
from kortex_api.RouterClientSendOptions import RouterClientSendOptions
from kortex_api.autogen.client_stubs.DeviceConfigClientRpc import DeviceConfigClient
from kortex_api.autogen.messages import Session_pb2, Common_pb2

import sys
import time
import grpc

# Replace this with your robot’s actual IP
IP_ADDRESS = "192.168.1.10"
PORT = 10000  # Default Kortex port

# Setup API session
def create_session():
    credentials = Session_pb2.CreateSessionInfo()
    credentials.username = "admin"
    credentials.password = "admin"

    router = RouterClient(IP_ADDRESS, PORT)
    router.connect()

    session_manager = SessionManager(router)
    session_manager.createSession(credentials)

    return router, session_manager

# Initialize and move to home
def main():
    router, session_manager = create_session()
    base = BaseClient(router)
    
    # Move the arm to home
    print("Moving to home position...")
    base.MoveHome()

    # Cleanup
    session_manager.closeSession()
    router.disconnect()

if __name__ == "__main__":
    main()
