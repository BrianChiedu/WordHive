#!/usr/bin/env python3
# server.py - Fixed implementation with better connection handling
import os
import time
import zmq
import json
import threading
import uuid
from collections import defaultdict, Counter
import logging
import argparse
import socket
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WordCountServer')

class WorkNode:
    """Represents a worker node in the distributed system."""
    def __init__(self, node_id, address):
        self.node_id = node_id
        self.address = address
        self.status = "idle"  # idle, busy, disconnected
        self.last_heartbeat = time.time()
        self.current_task = None
        self.performance_stats = {
            "completed_tasks": 0,
            "avg_processing_time": 0,
            "total_processing_time": 0
        }
    
    def update_heartbeat(self):
        """Update the last heartbeat time."""
        self.last_heartbeat = time.time()
    
    def assign_task(self, task_id):
        """Assign a task to this worker."""
        self.status = "busy"
        self.current_task = task_id
    
    def complete_task(self, processing_time):
        """Mark a task as completed and update performance stats."""
        self.status = "idle"
        self.current_task = None
        self.performance_stats["completed_tasks"] += 1
        
        # Update average processing time
        total_time = self.performance_stats["total_processing_time"] + processing_time
        self.performance_stats["total_processing_time"] = total_time
        self.performance_stats["avg_processing_time"] = total_time / self.performance_stats["completed_tasks"]
    
    def is_alive(self, timeout=15):
        """Check if the worker is still alive based on heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout


class DistributedFileProcessor:
    """
    Server component that manages worker nodes and distributes file processing tasks.
    """
    def __init__(self, server_address="*", 
                 client_port=5555, 
                 worker_port=5556,
                 heartbeat_port=5557):
        self.server_address = server_address
        self.client_port = client_port
        self.worker_port = worker_port
        self.heartbeat_port = heartbeat_port
        
        # ZeroMQ context
        self.context = zmq.Context()
        
        # Worker management
        self.workers = {}  # node_id -> WorkNode
        self.idle_workers = set()  # set of node_ids
        
        # Task management
        self.tasks = {}  # task_id -> task_info
        self.pending_tasks = []  # list of task_ids
        self.task_results = defaultdict(list)  # client_id -> [task_results]
        self.client_tasks = defaultdict(list)  # client_id -> [task_ids]
        
        # Processing state
        self.file_chunks = {}  # chunk_id -> (file_path, chunk_text)
        self.next_chunk_id = 0
        
        # Running state
        self.running = False
        self.threads = []
        
        # Signal handling
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle signals for graceful shutdown."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the server and worker management threads."""
        logger.info("Starting server...")
        self.running = True
        
        # Create and bind sockets
        self._create_sockets()
        
        # Start heartbeat listener thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        # Start worker message handler thread
        worker_thread = threading.Thread(target=self._worker_loop)
        worker_thread.daemon = True
        worker_thread.start()
        self.threads.append(worker_thread)
        
        # Start task scheduler thread
        scheduler_thread = threading.Thread(target=self._task_scheduler_loop)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        self.threads.append(scheduler_thread)
        
        # Start dead worker cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_loop)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        self.threads.append(cleanup_thread)
        
        # Start client handler in main thread
        logger.info(f"Server started on client port {self.client_port}, worker port {self.worker_port}")
        self._client_loop()
    
    def _create_sockets(self):
        """Create and bind ZeroMQ sockets."""
        # Client socket (REP)
        try:
            self.client_socket = self.context.socket(zmq.REP)
            client_bind = f"tcp://{self.server_address}:{self.client_port}"
            self.client_socket.bind(client_bind)
            logger.info(f"Client socket bound to {client_bind}")
        except zmq.error.ZMQError as e:
            logger.error(f"Failed to bind client socket: {e}")
            raise
        
        # Worker socket (ROUTER)
        try:
            self.worker_socket = self.context.socket(zmq.ROUTER)
            worker_bind = f"tcp://{self.server_address}:{self.worker_port}"
            self.worker_socket.bind(worker_bind)
            logger.info(f"Worker socket bound to {worker_bind}")
        except zmq.error.ZMQError as e:
            logger.error(f"Failed to bind worker socket: {e}")
            self.client_socket.close()
            raise
        
        # Heartbeat socket (SUB)
        try:
            self.heartbeat_socket = self.context.socket(zmq.SUB)
            heartbeat_bind = f"tcp://{self.server_address}:{self.heartbeat_port}"
            self.heartbeat_socket.bind(heartbeat_bind)
            self.heartbeat_socket.setsockopt_string(zmq.SUBSCRIBE, "heartbeat")
            logger.info(f"Heartbeat socket bound to {heartbeat_bind}")
        except zmq.error.ZMQError as e:
            logger.error(f"Failed to bind heartbeat socket: {e}")
            self.client_socket.close()
            self.worker_socket.close()
            raise
    
    def stop(self):
        """Stop the server and all threads."""
        logger.info("Stopping server...")
        self.running = False
        
        # Wait for threads to complete
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        # Close sockets with zero linger time to avoid hanging
        logger.debug("Closing sockets...")
        if hasattr(self, 'client_socket'):
            self.client_socket.close(linger=0)
        if hasattr(self, 'worker_socket'):
            self.worker_socket.close(linger=0)
        if hasattr(self, 'heartbeat_socket'):
            self.heartbeat_socket.close(linger=0)
        
        # Terminate context
        logger.debug("Terminating ZeroMQ context...")
        self.context.term()
        
        logger.info("Server stopped")
    
    def _heartbeat_loop(self):
        """Listen for heartbeats from workers."""
        logger.info("Heartbeat listener started")
        
        # Create poller for heartbeat socket
        poller = zmq.Poller()
        poller.register(self.heartbeat_socket, zmq.POLLIN)
        
        while self.running:
            try:
                # Poll with timeout to allow for clean shutdown
                socks = dict(poller.poll(timeout=100))
                
                if self.heartbeat_socket in socks and socks[self.heartbeat_socket] == zmq.POLLIN:
                    message = self.heartbeat_socket.recv_multipart()
                    
                    # Parse heartbeat message
                    if len(message) >= 2:
                        _, worker_info = message  # topic, data
                        try:
                            worker_data = json.loads(worker_info.decode('utf-8'))
                            
                            node_id = worker_data.get('node_id')
                            address = worker_data.get('address')
                            
                            # Update existing worker or register new one
                            if node_id in self.workers:
                                self.workers[node_id].update_heartbeat()
                                
                                # Check if worker status changed from busy to idle
                                if (self.workers[node_id].status == "busy" and 
                                    worker_data.get('status') == "idle"):
                                    self.workers[node_id].status = "idle"
                                    self.idle_workers.add(node_id)
                                    
                            else:
                                # Register new worker
                                logger.info(f"New worker connected via heartbeat: {node_id} at {address}")
                                self.workers[node_id] = WorkNode(node_id, address)
                                self.idle_workers.add(node_id)
                        except json.JSONDecodeError:
                            logger.warning("Received malformed heartbeat message")
                
            except zmq.error.ZMQError as e:
                logger.error(f"ZMQ error in heartbeat loop: {str(e)}")
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}", exc_info=True)
            
            # Brief pause to prevent high CPU usage
            time.sleep(0.01)
    
    def _cleanup_loop(self):
        """Periodically check for dead workers and clean them up."""
        logger.info("Worker cleanup loop started")
        
        while self.running:
            try:
                # Check for dead workers
                dead_workers = []
                for node_id, worker in self.workers.items():
                    if not worker.is_alive():
                        dead_workers.append(node_id)
                
                # Remove dead workers
                for node_id in dead_workers:
                    logger.warning(f"Worker {node_id} appears to be dead, removing")
                    if node_id in self.idle_workers:
                        self.idle_workers.remove(node_id)
                    
                    # Handle tasks assigned to dead workers
                    if self.workers[node_id].current_task:
                        task_id = self.workers[node_id].current_task
                        logger.warning(f"Rescheduling task {task_id} from dead worker {node_id}")
                        self.pending_tasks.append(task_id)
                    
                    del self.workers[node_id]
                
                # Log worker status periodically
                if dead_workers:
                    logger.info(f"Current workers: {len(self.workers)}, Idle: {len(self.idle_workers)}")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}", exc_info=True)
            
            # Sleep between checks
            time.sleep(5.0)
    
    def _worker_loop(self):
        """Handle messages from workers."""
        logger.info("Worker handler started")
        
        # Create poller for worker socket
        poller = zmq.Poller()
        poller.register(self.worker_socket, zmq.POLLIN)
        
        while self.running:
            try:
                # Poll with timeout to allow for clean shutdown
                socks = dict(poller.poll(timeout=100))
                
                if self.worker_socket in socks and socks[self.worker_socket] == zmq.POLLIN:
                    # Receive message parts: [worker_id, empty, message]
                    message_parts = self.worker_socket.recv_multipart()
                    
                    if len(message_parts) >= 2:
                        worker_id = message_parts[0]
                        # Support both formats: with or without empty delimiter frame
                        data_frame = message_parts[2] if len(message_parts) >= 3 else message_parts[1]
                        
                        try:
                            message = json.loads(data_frame.decode('utf-8'))
                            
                            message_type = message.get('type')
                            
                            if message_type == 'task_result':
                                self._handle_task_result(worker_id, message)
                            elif message_type == 'register':
                                self._handle_worker_registration(worker_id, message)
                            elif message_type == 'status_update':
                                self._handle_worker_status_update(worker_id, message)
                        except json.JSONDecodeError:
                            logger.warning(f"Received malformed message from worker {worker_id.decode('utf-8')}")
                
            except zmq.error.ZMQError as e:
                logger.error(f"ZMQ error in worker loop: {str(e)}")
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}", exc_info=True)
            
            # Brief pause to prevent high CPU usage
            time.sleep(0.01)
    
    def _handle_task_result(self, worker_id, message):
        """Process task results from a worker."""
        task_id = message.get('task_id')
        client_id = message.get('client_id')
        processing_time = message.get('processing_time', 0)
        result = message.get('result', {})
        success = message.get('success', True)
        
        worker_id_str = worker_id.decode('utf-8')
        logger.info(f"Received result for task {task_id} from worker {worker_id_str}")
        
        # Update worker status
        if worker_id_str in self.workers:
            self.workers[worker_id_str].complete_task(processing_time)
            self.idle_workers.add(worker_id_str)
        
        # Only store successful results
        if success and client_id in self.client_tasks and task_id in self.client_tasks[client_id]:
            self.task_results[client_id].append(result)
            
            # Check if all tasks for this client are complete
            remaining_tasks = sum(1 for t in self.client_tasks[client_id] if t in self.tasks)
            
            if remaining_tasks == 0:
                logger.info(f"All tasks complete for client {client_id}")
        
        # Remove task from tracking
        if task_id in self.tasks:
            del self.tasks[task_id]
        
        # Acknowledge receipt to worker
        try:
            response = {
                'type': 'task_ack',
                'task_id': task_id
            }
            self.worker_socket.send_multipart([worker_id, b'', json.dumps(response).encode('utf-8')])
        except Exception as e:
            logger.error(f"Error sending task acknowledgment: {str(e)}")
    
    def _handle_worker_registration(self, worker_id, message):
        """Handle worker registration."""
        worker_id_str = worker_id.decode('utf-8')
        address = message.get('address', 'unknown')
        
        logger.info(f"Worker registration from {worker_id_str} at {address}")
        
        # Create or update worker
        if worker_id_str not in self.workers:
            self.workers[worker_id_str] = WorkNode(worker_id_str, address)
            self.idle_workers.add(worker_id_str)
        else:
            self.workers[worker_id_str].address = address
            self.workers[worker_id_str].update_heartbeat()
            self.workers[worker_id_str].status = "idle"
            self.idle_workers.add(worker_id_str)
        
        # Acknowledge registration
        try:
            response = {
                'type': 'register_ack',
                'server_time': time.time()
            }
            self.worker_socket.send_multipart([worker_id, b'', json.dumps(response).encode('utf-8')])
        except Exception as e:
            logger.error(f"Error sending registration acknowledgment: {str(e)}")
    
    def _handle_worker_status_update(self, worker_id, message):
        """Handle worker status updates."""
        worker_id_str = worker_id.decode('utf-8')
        status = message.get('status')
        
        if worker_id_str in self.workers:
            old_status = self.workers[worker_id_str].status
            self.workers[worker_id_str].status = status
            self.workers[worker_id_str].update_heartbeat()
            
            # Update idle workers set
            if status == "idle" and old_status != "idle":
                self.idle_workers.add(worker_id_str)
            elif status != "idle" and worker_id_str in self.idle_workers:
                self.idle_workers.remove(worker_id_str)
        
        # Acknowledge status update
        try:
            response = {
                'type': 'status_ack'
            }
            self.worker_socket.send_multipart([worker_id, b'', json.dumps(response).encode('utf-8')])
        except Exception as e:
            logger.error(f"Error sending status acknowledgment: {str(e)}")
    
    def _task_scheduler_loop(self):
        """Schedule pending tasks to available workers."""
        logger.info("Task scheduler started")
        
        while self.running:
            try:
                # Check if there are pending tasks and idle workers
                if self.pending_tasks and self.idle_workers:
                    task_id = self.pending_tasks.pop(0)
                    
                    if task_id in self.tasks:
                        # Get task details
                        task = self.tasks[task_id]
                        
                        # Get an idle worker
                        if self.idle_workers:
                            node_id = next(iter(self.idle_workers))
                            self.idle_workers.remove(node_id)
                            
                            # Assign task to worker
                            worker = self.workers[node_id]
                            worker.assign_task(task_id)
                            
                            # Send task to worker
                            logger.info(f"Assigning task {task_id} to worker {node_id}")
                            worker_id = node_id.encode('utf-8')
                            
                            try:
                                task_message = {
                                    'type': 'task',
                                    'task_id': task_id,
                                    'client_id': task['client_id'],
                                    'chunk_text': task['chunk_text'],
                                    'chunk_id': task['chunk_id'],
                                    'file_path': task['file_path']
                                }
                                
                                self.worker_socket.send_multipart([
                                    worker_id, 
                                    b'', 
                                    json.dumps(task_message).encode('utf-8')
                                ])
                            except Exception as e:
                                logger.error(f"Error sending task to worker: {str(e)}")
                                # Put task back in queue
                                self.pending_tasks.insert(0, task_id)
                                # Worker might be dead, mark it busy to avoid reuse
                                worker.status = "busy"
                        else:
                            # No idle workers, put task back in queue
                            self.pending_tasks.insert(0, task_id)
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}", exc_info=True)
            
            # Brief pause to prevent high CPU usage
            time.sleep(0.05)
    
    def _client_loop(self):
        """Handle client requests in the main thread."""
        logger.info("Client handler started")
        
        # Create poller for client socket
        poller = zmq.Poller()
        poller.register(self.client_socket, zmq.POLLIN)
        
        while self.running:
            try:
                # Poll with timeout to allow for clean shutdown
                socks = dict(poller.poll(timeout=100))
                
                if self.client_socket in socks and socks[self.client_socket] == zmq.POLLIN:
                    # Receive client message
                    message_data = self.client_socket.recv()
                    
                    try:
                        message = json.loads(message_data.decode('utf-8'))
                        
                        message_type = message.get('type')
                        client_id = message.get('client_id')
                        
                        if message_type == 'process_files':
                            response = self._handle_process_files(client_id, message)
                        elif message_type == 'get_results':
                            response = self._handle_get_results(client_id)
                        elif message_type == 'get_worker_status':
                            response = self._handle_get_worker_status()
                        else:
                            response = {
                                'status': 'error',
                                'message': f'Unknown message type: {message_type}'
                            }
                        
                        # Send response to client
                        self.client_socket.send(json.dumps(response).encode('utf-8'))
                    
                    except json.JSONDecodeError:
                        # Send error response for malformed JSON
                        error_response = {
                            'status': 'error',
                            'message': 'Malformed JSON request'
                        }
                        self.client_socket.send(json.dumps(error_response).encode('utf-8'))
                
            except zmq.error.ZMQError as e:
                logger.error(f"ZMQ error in client loop: {str(e)}")
            except Exception as e:
                logger.error(f"Error in client loop: {str(e)}", exc_info=True)
                
                # Send error response to client
                try:
                    error_response = {
                        'status': 'error',
                        'message': str(e)
                    }
                    self.client_socket.send(json.dumps(error_response).encode('utf-8'))
                except:
                    pass
            
            # Brief pause to prevent high CPU usage
            time.sleep(0.01)
    
    def _handle_process_files(self, client_id, message):
        """Handle file processing request from client."""
        file_data = message.get('file_data', [])
        
        if not file_data:
            return {
                'status': 'error',
                'message': 'No file data provided'
            }
        
        logger.info(f"Processing {len(file_data)} files for client {client_id}")
        
        # Clear previous results for this client
        if client_id in self.task_results:
            self.task_results[client_id] = []
        
        # Clear previous tasks for this client
        if client_id in self.client_tasks:
            # Remove old tasks from the task list
            for task_id in self.client_tasks[client_id]:
                if task_id in self.tasks:
                    del self.tasks[task_id]
                
                # Remove from pending tasks if present
                if task_id in self.pending_tasks:
                    self.pending_tasks.remove(task_id)
            
            self.client_tasks[client_id] = []
        
        # Process each file
        task_ids = []
        
        for file_entry in file_data:
            file_path = file_entry.get('path')
            file_content = file_entry.get('content')
            
            # Split content into chunks for processing
            chunks = self._split_content(file_content, chunk_size=1024*1024)  # 1MB chunks
            
            for i, chunk_text in enumerate(chunks):
                # Create a task for this chunk
                task_id = str(uuid.uuid4())
                chunk_id = self.next_chunk_id
                self.next_chunk_id += 1
                
                task = {
                    'task_id': task_id,
                    'client_id': client_id,
                    'file_path': file_path,
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_text
                }
                
                # Store task
                self.tasks[task_id] = task
                self.client_tasks[client_id].append(task_id)
                task_ids.append(task_id)
                
                # Add to pending tasks
                self.pending_tasks.append(task_id)
                
                # Store file chunk mapping
                self.file_chunks[chunk_id] = (file_path, chunk_text)
        
        return {
            'status': 'success',
            'message': f'Processing {len(task_ids)} chunks across {len(file_data)} files',
            'task_count': len(task_ids)
        }
    
    def _handle_get_results(self, client_id):
        """Handle request for processing results from client."""
        if client_id not in self.client_tasks:
            return {
                'status': 'error',
                'message': 'No tasks found for this client'
            }
        
        # Check if all tasks are complete
        remaining_tasks = sum(1 for t in self.client_tasks[client_id] if t in self.tasks)
        
        if remaining_tasks > 0:
            return {
                'status': 'in_progress',
                'message': f'Processing in progress. {remaining_tasks} tasks remaining.',
                'remaining': remaining_tasks,
                'total': len(self.client_tasks[client_id])
            }
        
        # All tasks complete, aggregate results
        combined_results = Counter()
        
        for result in self.task_results[client_id]:
            word_counts = result.get('word_counts', {})
            for word, count in word_counts.items():
                combined_results[word] += count
        
        # Calculate processing details
        total_words = sum(combined_results.values())
        unique_words = len(combined_results)
        
        # Return top words in the results
        top_words = dict(combined_results.most_common(50))
        
        return {
            'status': 'complete',
            'message': 'Processing complete',
            'results': {
                'total_words': total_words,
                'unique_words': unique_words,
                'top_words': top_words,
                'all_counts': dict(combined_results)
            }
        }
    
    def _handle_get_worker_status(self):
        """Handle request for worker status information."""
        worker_info = []
        
        for node_id, worker in self.workers.items():
            worker_info.append({
                'node_id': node_id,
                'address': worker.address,
                'status': worker.status,
                'last_heartbeat': worker.last_heartbeat,
                'current_task': worker.current_task,
                'stats': worker.performance_stats
            })
        
        return {
            'status': 'success',
            'worker_count': len(self.workers),
            'idle_workers': len(self.idle_workers),
            'workers': worker_info,
            'pending_tasks': len(self.pending_tasks)
        }
    
    def _split_content(self, content, chunk_size=1024*1024):
        """Split content into chunks."""
        if not content:
            return []
        
        # Simple splitting by size
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i+chunk_size])
        
        return chunks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed Word Count Server')
    parser.add_argument('--host', default='*', help='Server host address')
    parser.add_argument('--client-port', type=int, default=5555, help='Port for client connections')
    parser.add_argument('--worker-port', type=int, default=5556, help='Port for worker connections')
    parser.add_argument('--heartbeat-port', type=int, default=5557, help='Port for worker heartbeats')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger('WordCountServer').setLevel(logging.DEBUG)
    
    # Create and start server
    server = DistributedFileProcessor(
        server_address=args.host,
        client_port=args.client_port,
        worker_port=args.worker_port,
        heartbeat_port=args.heartbeat_port
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("Interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Server failed with error: {str(e)}", exc_info=True)
    finally:
        server.stop()