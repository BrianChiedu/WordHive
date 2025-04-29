#!/usr/bin/env python3
# worker.py - Fixed implementation with better connection handling
import os
import re
import time
import zmq
import json
import threading
import uuid
from collections import Counter
import logging
import argparse
import socket
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WordCountWorker')

class WordCountWorker:
    """
    Worker node that processes text chunks and reports results back to server.
    """
    def __init__(self, server_address, worker_port=5556, heartbeat_port=5557, heartbeat_interval=5):
        self.server_address = server_address
        self.worker_port = worker_port
        self.heartbeat_port = heartbeat_port
        self.heartbeat_interval = heartbeat_interval
        
        # Create node ID (unique identifier for this worker)
        hostname = socket.gethostname()
        self.node_id = f"worker-{hostname}-{uuid.uuid4().hex[:8]}"
        
        # Get worker address
        try:
            # Try to get actual network IP instead of localhost
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # This doesn't actually establish a connection, but helps get the right interface
            s.connect((server_address, 1))  
            self.worker_address = s.getsockname()[0]
            s.close()
        except:
            self.worker_address = "127.0.0.1"
        
        # ZeroMQ context
        self.context = zmq.Context()
        
        # Connection settings
        self.reconnect_interval_min = 1000  # ms
        self.reconnect_interval_max = 32000  # ms
        self.reconnect_interval = self.reconnect_interval_min
        
        # Worker state
        self.status = "initializing"  # initializing, idle, busy, shutting_down
        self.current_task = None
        self.running = False
        self.threads = []
        
        # Performance tracking
        self.completed_tasks = 0
        self.total_processing_time = 0
        
        logger.info(f"Worker initialized with ID: {self.node_id}")
        logger.info(f"Worker address: {self.worker_address}")
    
    def start(self):
        """Start the worker and its threads."""
        self.running = True
        
        # Create sockets
        self._create_sockets()
        
        # Register with server
        if not self._register_with_server():
            logger.error("Failed to register with server. Exiting.")
            self.running = False
            return False
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        # Start main worker loop
        self._worker_loop()
        return True
    
    def stop(self):
        """Stop the worker and its threads."""
        logger.info("Stopping worker...")
        self.status = "shutting_down"
        self.running = False
        
        # Send final heartbeat notifying shutdown
        self._send_heartbeat()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Close sockets
        try:
            self.server_socket.close(linger=0)
            self.heartbeat_socket.close(linger=0)
        except:
            pass
        
        # Terminate context
        try:
            self.context.term()
        except:
            pass
        
        logger.info("Worker stopped")
    
    def _create_sockets(self):
        """Create and configure the ZeroMQ sockets."""
        # Server socket (DEALER) for task communication
        self.server_socket = self.context.socket(zmq.DEALER)
        self.server_socket.setsockopt(zmq.IDENTITY, self.node_id.encode('utf-8'))
        
        # Configure socket timeouts and reconnect intervals
        self.server_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        self.server_socket.setsockopt(zmq.SNDTIMEO, 5000)  # 5 second timeout
        self.server_socket.setsockopt(zmq.RECONNECT_IVL, self.reconnect_interval_min)
        self.server_socket.setsockopt(zmq.RECONNECT_IVL_MAX, self.reconnect_interval_max)
        
        # Connect to server
        server_connection = f"tcp://{self.server_address}:{self.worker_port}"
        self.server_socket.connect(server_connection)
        logger.info(f"Connected to server at {server_connection}")
        
        # Heartbeat socket (PUB)
        self.heartbeat_socket = self.context.socket(zmq.PUB)
        heartbeat_connection = f"tcp://{self.server_address}:{self.heartbeat_port}"
        self.heartbeat_socket.connect(heartbeat_connection)
        logger.info(f"Publishing heartbeats to {heartbeat_connection}")
    
    def _register_with_server(self):
        """Register this worker with the server."""
        logger.info(f"Registering with server as {self.node_id}")
        
        registration = {
            'type': 'register',
            'node_id': self.node_id,
            'address': self.worker_address,
            'hostname': socket.gethostname()
        }
        
        # Send registration message (with retry)
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries and self.running:
            try:
                # Send registration message
                self.server_socket.send_multipart([b'', json.dumps(registration).encode('utf-8')])
                
                # Wait for acknowledgment
                if self.server_socket.poll(timeout=5000):  # 5 second timeout
                    try:
                        frames = self.server_socket.recv_multipart()
                        
                        # Handle response based on frame structure
                        if len(frames) >= 1:
                            response_frame = frames[-1]  # Last frame contains the response
                            response = json.loads(response_frame.decode('utf-8'))
                            
                            if response.get('type') == 'register_ack':
                                logger.info("Registration acknowledged by server")
                                self.status = "idle"
                                return True
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in registration response: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error processing registration response: {str(e)}")
            
                logger.warning(f"Registration attempt {retry_count + 1} failed, retrying...")
                retry_count += 1
                time.sleep(1)  # Wait before retrying
                
            except Exception as e:
                logger.error(f"Error during registration attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                time.sleep(1)  # Wait before retrying
        
        return False
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to the server."""
        logger.info(f"Starting heartbeat loop with interval {self.heartbeat_interval}s")
        
        while self.running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                time.sleep(1)  # Short delay before retry
    
    def _send_heartbeat(self):
        """Send a heartbeat message to the server."""
        try:
            heartbeat = {
                'node_id': self.node_id,
                'status': self.status,
                'address': self.worker_address,
                'timestamp': time.time(),
                'current_task': self.current_task,
                'completed_tasks': self.completed_tasks
            }
            
            self.heartbeat_socket.send_multipart([
                b'heartbeat',
                json.dumps(heartbeat).encode('utf-8')
            ])
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {str(e)}")
    
    def _worker_loop(self):
        """Main worker loop to process tasks."""
        logger.info("Starting main worker loop")
        
        poller = zmq.Poller()
        poller.register(self.server_socket, zmq.POLLIN)
        
        while self.running:
            try:
                # Use polling with timeout to allow for clean shutdown
                socks = dict(poller.poll(timeout=1000))  # 1 second timeout
                
                if self.server_socket in socks and socks[self.server_socket] == zmq.POLLIN:
                    # Receive message frames, handle different frame structures
                    frames = self.server_socket.recv_multipart()
                    
                    # Parse message - the message should be in the last frame
                    if len(frames) > 0:
                        try:
                            message_data = frames[-1]  # Last frame contains the message
                            message = json.loads(message_data.decode('utf-8'))
                            message_type = message.get('type')
                            
                            if message_type == 'task':
                                self._process_task(message)
                            elif message_type == 'task_ack':
                                logger.debug(f"Server acknowledged result for task {message.get('task_id')}")
                            elif message_type == 'status_ack':
                                logger.debug("Server acknowledged status update")
                            elif message_type == 'shutdown':
                                logger.info("Received shutdown command from server")
                                self.running = False
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON message from server: {str(e)}")
                        except Exception as e:
                            logger.error(f"Error processing message: {str(e)}")
                
            except zmq.error.Again:
                # Socket timeout, just continue
                pass
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                # Brief pause to avoid tight error loop
                time.sleep(0.1)
    
    def _process_task(self, task_message):
        """Process a text processing task."""
        task_id = task_message.get('task_id')
        client_id = task_message.get('client_id')
        chunk_text = task_message.get('chunk_text', '')
        chunk_id = task_message.get('chunk_id')
        file_path = task_message.get('file_path', '')
        
        logger.info(f"Processing task {task_id} (chunk {chunk_id} from {file_path})")
        
        # Update status
        self.status = "busy"
        self.current_task = task_id
        
        # Send status update
        try:
            status_update = {
                'type': 'status_update',
                'node_id': self.node_id,
                'status': self.status,
                'current_task': task_id
            }
            
            self.server_socket.send_multipart([b'', json.dumps(status_update).encode('utf-8')])
        except Exception as e:
            logger.warning(f"Failed to send status update: {str(e)}")
        
        # Process the chunk
        start_time = time.time()
        
        try:
            result = self._count_words(chunk_text)
            success = True
        except Exception as e:
            logger.error(f"Error processing task {task_id}: {str(e)}")
            result = {}
            success = False
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update performance stats
        if success:
            self.completed_tasks += 1
            self.total_processing_time += processing_time
        
        # Send result back to server
        try:
            result_message = {
                'type': 'task_result',
                'task_id': task_id,
                'client_id': client_id,
                'chunk_id': chunk_id,
                'file_path': file_path,
                'processing_time': processing_time,
                'success': success,
                'result': {
                    'word_counts': result
                }
            }
            
            self.server_socket.send_multipart([b'', json.dumps(result_message).encode('utf-8')])
            logger.info(f"Sent result for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to send task result: {str(e)}")
        
        # Update status
        self.status = "idle"
        self.current_task = None
        
        # Send status update
        try:
            status_update = {
                'type': 'status_update',
                'node_id': self.node_id,
                'status': self.status,
                'current_task': None
            }
            
            self.server_socket.send_multipart([b'', json.dumps(status_update).encode('utf-8')])
        except Exception as e:
            logger.warning(f"Failed to send status update: {str(e)}")
        
        logger.info(f"Completed task {task_id} in {processing_time:.2f}s")
    
    def _count_words(self, text):
        """Count word frequencies in text."""
        if not text:
            return {}
        
        # Remove punctuation, normalize whitespace, and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split by whitespace
        words_list = cleaned_text.split()
        
        # Filter to ensure only actual words are counted (at least 2 characters)
        valid_words = [word for word in words_list if re.match(r'^[a-z][a-z0-9_]*$', word) and len(word) >= 2]
        
        # Count words
        word_counts = Counter(valid_words)
        
        return dict(word_counts)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed Word Count Worker')
    parser.add_argument('--server', required=True, help='Server address')
    parser.add_argument('--worker-port', type=int, default=5556, help='Server port for worker connections')
    parser.add_argument('--heartbeat-port', type=int, default=5557, help='Server port for heartbeats')
    parser.add_argument('--heartbeat-interval', type=int, default=5, help='Heartbeat interval in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger('WordCountWorker').setLevel(logging.DEBUG)
    
    # Create and start worker
    worker = WordCountWorker(
        server_address=args.server,
        worker_port=args.worker_port,
        heartbeat_port=args.heartbeat_port,
        heartbeat_interval=args.heartbeat_interval
    )
    
    try:
        success = worker.start()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Worker failed with error: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        worker.stop()