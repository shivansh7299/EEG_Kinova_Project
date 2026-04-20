import threading

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

# Replace this with your robot’s actual IP
IP_ADDRESS = "192.168.1.10"
PORT = 10000  # Default Kortex port

# Setup API session
def create_session():
    credentials = Session_pb2.CreateSessionInfo()
    credentials.username = "admin"
    credentials.password = "admin"
    credentials.session_inactivity_timeout = 60000
    credentials.connection_inactivity_timeout = 2000

    transport = TCPTransport()
    transport.connect(IP_ADDRESS, PORT)

    # RouterClient expects a transport object, not IP/port strings.
    router = RouterClient(transport, lambda e: print("Kortex error:", e))

    session_manager = SessionManager(router)
    session_manager.CreateSession(credentials)

    return transport, router, session_manager


def move_to_saved_home(base, timeout=30):
    """Execute robot's saved Home action if it exists."""
    req = Base_pb2.RequestedActionType()
    req.action_type = Base_pb2.REACH_JOINT_ANGLES
    actions = base.ReadAllActions(req)

    home_handle = None
    for action in actions.action_list:
        if action.name.strip().lower() == "home":
            home_handle = action.handle
            break

    if home_handle is None:
        print("No predefined 'Home' action found on the robot.")
        print("Define a Home action on the Kinova device, or use a joint-angle action script.")
        return False

    done = threading.Event()

    def callback(notification, event=done):
        if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
            event.set()

    handle = base.OnNotificationActionTopic(callback, Base_pb2.NotificationOptions())
    base.ExecuteActionFromReference(home_handle)
    ok = done.wait(timeout)
    base.Unsubscribe(handle)

    if not ok:
        print(f"Home action timed out after {timeout}s.")
    return ok

# Initialize and move to home
def main():
    transport = None
    session_manager = None
    try:
        transport, router, session_manager = create_session()
        base = BaseClient(router)

        # Match original intent: move to the robot's predefined Home position.
        print("Moving to home position...")
        if move_to_saved_home(base):
            print("Home action complete.")
    finally:
        if session_manager is not None:
            try:
                session_manager.CloseSession()
            except Exception as e:
                print(f"Session close warning: {e}")
        if transport is not None:
            try:
                transport.disconnect()
            except Exception as e:
                print(f"Transport disconnect warning: {e}")

if __name__ == "__main__":
    main()
