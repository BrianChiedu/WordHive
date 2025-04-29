#!/usr/bin/env python3
# connection_test.py
"""
Simple test script to verify connectivity to the server.
"""
import sys
import zmq
import json
import time

def test_server_connection(server_address="localhost", client_port=5555):
    """Test basic connectivity to the server."""
    print(f"Testing connection to server at {server_address}:{client_port}...")
    
    # Create ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    # Connect to server
    try:
        socket.connect(f"tcp://{server_address}:{client_port}")
        print(f"Socket connected to tcp://{server_address}:{client_port}")
    except Exception as e:
        print(f"Error connecting to socket: {e}")
        return False
    
    # Create a simple request
    request = {
        'type': 'get_worker_status',
        'client_id': 'test-client'
    }
    
    # Set a timeout
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 seconds timeout
    
    try:
        # Send request
        print("Sending request...")
        socket.send(json.dumps(request).encode('utf-8'))
        print("Request sent, waiting for response...")
        
        # Wait for response
        response_data = socket.recv()
        response = json.loads(response_data.decode('utf-8'))
        
        print("Response received!")
        print(f"Response status: {response.get('status')}")
        print(f"Worker count: {response.get('worker_count', 0)}")
        
        socket.close()
        context.term()
        
        return True
        
    except zmq.error.Again:
        print("Timeout waiting for response. Is the server running?")
        socket.close()
        context.term()
        return False
        
    except Exception as e:
        print(f"Error during communication: {e}")
        socket.close()
        context.term()
        return False

if __name__ == "__main__":
    server = "localhost"
    port = 5555
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        server = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    # Run test
    test_server_connection(server, port)