#!/usr/bin/env python3
# worker_manager.py
"""
Utility for starting and managing worker nodes in the distributed word count system.
"""
import os
import sys
import argparse
import subprocess
import time
import signal
import socket
import json
import zmq
import logging
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WorkerManager')

class WorkerManager:
    """Manages worker processes."""
    def __init__(self, server_address, worker_port=5556, heartbeat_port=5557):
        self.server_address = server_address
        self.worker_port = worker_port
        self.heartbeat_port = heartbeat_port
        self.workers = {}  # pid -> process_info
    
    def start_workers(self, count, worker_script='worker.py'):
        """Start a number of worker processes."""
        logger.info(f"Starting {count} worker processes")
        
        # Check if worker script exists
        if not os.path.exists(worker_script):
            logger.error(f"Worker script '{worker_script}' not found")
            return False
        
        # Start workers
        for i in range(count):
            self._start_worker(worker_script)
        
        return True
    
    def _start_worker(self, worker_script):
        """Start a single worker process."""
        try:
            # Build command
            cmd = [
                sys.executable,
                worker_script,
                '--server', self.server_address,
                '--worker-port', str(self.worker_port),
                '--heartbeat-port', str(self.heartbeat_port)
            ]
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store process info
            self.workers[process.pid] = {
                'process': process,
                'started': time.time(),
                'status': 'starting'
            }
            
            logger.info(f"Started worker process with PID {process.pid}")
            return process.pid
            
        except Exception as e:
            logger.error(f"Error starting worker: {str(e)}")
            return None
    
    def stop_workers(self):
        """Stop all worker processes."""
        logger.info(f"Stopping {len(self.workers)} worker processes")
        
        for pid, worker_info in list(self.workers.items()):
            self._stop_worker(pid)
    
    def _stop_worker(self, pid):
        """Stop a single worker process."""
        if pid not in self.workers:
            logger.warning(f"Worker with PID {pid} not found")
            return False
        
        try:
            process = self.workers[pid]['process']
            
            # Send SIGTERM to process
            process.terminate()
            
            # Wait for process to exit (with timeout)
            try:
                process.wait(timeout=5)
                logger.info(f"Worker process {pid} stopped")
            except subprocess.TimeoutExpired:
                # Force kill
                process.kill()
                logger.warning(f"Worker process {pid} force killed")
            
            # Remove from workers
            del self.workers[pid]
            return True
            
        except Exception as e:
            logger.error(f"Error stopping worker {pid}: {str(e)}")
            return False
    
    def check_worker_status(self):
        """Check status of worker processes."""
        # Check if processes are still running
        for pid, worker_info in list(self.workers.items()):
            process = worker_info['process']
            
            # Check if process has exited
            exit_code = process.poll()
            if exit_code is not None:
                # Process has exited
                logger.warning(f"Worker process {pid} exited with code {exit_code}")
                
                # Read stderr for error message
                stderr = process.stderr.read()
                if stderr:
                    logger.error(f"Worker {pid} error output: {stderr}")
                
                # Remove from workers
                del self.workers[pid]
    
    def get_worker_status_from_server(self):
        """Get worker status information from server."""
        try:
            # Create ZeroMQ client
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            
            # Set timeout
            socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
            
            # Connect to server - client port is worker_port - 1
            client_port = self.worker_port - 1
            server_url = f"tcp://{self.server_address}:{client_port}"
            logger.info(f"Connecting to server at {server_url}")
            socket.connect(server_url)
            
            # Generate a unique client ID
            client_id = f"worker_manager-{os.getpid()}"
            
            # Send worker status request
            request = {
                'type': 'get_worker_status',
                'client_id': client_id
            }
            
            logger.debug(f"Sending request: {request}")
            socket.send(json.dumps(request).encode('utf-8'))
            
            # Wait for response
            try:
                response_data = socket.recv()
                
                # Try to decode as JSON - if it fails, treat as plain text
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    
                    # Verify response is a dictionary
                    if not isinstance(response, dict):
                        logger.warning(f"Response is not a dictionary: {response}")
                        response = {"status": "error", "message": f"Invalid response format: {response}"}
                        
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to decode JSON response: {e}")
                    # Store raw response for debugging
                    response = {"status": "error", "message": f"Invalid response: {response_data}"}
                
            except zmq.error.Again:
                logger.warning("Request for worker status timed out")
                response = {"status": "error", "message": "Request timed out"}
                
            except Exception as e:
                logger.error(f"Error receiving response: {e}")
                response = {"status": "error", "message": str(e)}
            
            # Close socket and context
            socket.close(linger=0)
            context.term()
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting worker status from server: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def print_status(self):
        """Print status of worker processes."""
        # Local process status
        print("\n=== Local Worker Processes ===")
        
        if not self.workers:
            print("No local worker processes")
        else:
            # Check for updates
            self.check_worker_status()
            
            # Create table
            table_data = []
            
            for pid, worker_info in self.workers.items():
                process = worker_info['process']
                started = worker_info['started']
                status = "Running" if process.poll() is None else f"Exited ({process.poll()})"
                uptime = time.time() - started
                
                table_data.append([pid, status, f"{uptime:.1f}s"])
            
            # Print table
            print(tabulate(table_data, headers=["PID", "Status", "Uptime"]))
        
        # Server-side worker status
        print("\n=== Server-side Worker Status ===")
        
        status_response = self.get_worker_status_from_server()
        
        # Make sure we have a dictionary
        if not isinstance(status_response, dict):
            print(f"Error: Unexpected response format: {status_response}")
            return
        
        # Check response status
        if status_response.get('status') != 'success':
            error_msg = status_response.get('message', 'Unknown error')
            print(f"Could not get status from server: {error_msg}")
            return
        
        # Get worker information
        workers = status_response.get('workers', [])
        
        if not workers:
            print("No workers connected to server")
        else:
            # Create table
            table_data = []
            
            for worker in workers:
                # Skip if not a dictionary
                if not isinstance(worker, dict):
                    continue
                    
                node_id = worker.get('node_id', 'unknown')
                address = worker.get('address', 'unknown')
                status = worker.get('status', 'unknown')
                
                # Get stats
                stats = worker.get('stats', {})
                if isinstance(stats, dict):
                    completed_tasks = stats.get('completed_tasks', 0)
                    avg_time = stats.get('avg_processing_time', 0)
                else:
                    completed_tasks = 0
                    avg_time = 0
                
                # Get last heartbeat time
                last_heartbeat = worker.get('last_heartbeat', 0)
                try:
                    time_since_heartbeat = time.time() - float(last_heartbeat)
                except (TypeError, ValueError):
                    time_since_heartbeat = 0
                
                table_data.append([
                    node_id, 
                    address, 
                    status, 
                    completed_tasks,
                    f"{avg_time:.2f}s" if avg_time else "N/A",
                    f"{time_since_heartbeat:.1f}s"
                ])
            
            # Print table
            print(tabulate(table_data, headers=[
                "Node ID", "Address", "Status", "Tasks", "Avg Time", "Last Heartbeat"
            ]))
            
            # Print summary
            worker_count = status_response.get('worker_count', 0)
            idle_workers = status_response.get('idle_workers', 0)
            pending_tasks = status_response.get('pending_tasks', 0)
            
            print(f"\nTotal workers: {worker_count}")
            print(f"Idle workers: {idle_workers}")
            print(f"Pending tasks: {pending_tasks}")
        
        print()


def main():
    """Main entry point."""
    # Parse command line arguments - simplified version
    parser = argparse.ArgumentParser(description='Distributed Word Count Worker Manager')
    
    # Common parameters
    parser.add_argument('--server', default='localhost', help='Server address')
    parser.add_argument('--worker-port', type=int, default=5556, help='Server worker port')
    parser.add_argument('--heartbeat-port', type=int, default=5557, help='Server heartbeat port')
    parser.add_argument('--worker-script', default='worker.py', help='Path to worker script')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    # Command and count
    parser.add_argument('command', nargs='?', default='status', help='Command: start, stop, status, monitor')
    parser.add_argument('count', nargs='?', type=int, default=1, help='Number of workers (for start command)')
    parser.add_argument('--interval', type=int, default=5, help='Update interval (for monitor command)')
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger('WorkerManager').setLevel(logging.DEBUG)
    
    # Create manager
    manager = WorkerManager(
        server_address=args.server,
        worker_port=args.worker_port,
        heartbeat_port=args.heartbeat_port
    )
    
    # Execute command
    if args.command == 'start':
        manager.start_workers(args.count, args.worker_script)
        manager.print_status()
        
    elif args.command == 'stop':
        manager.stop_workers()
        
    elif args.command == 'status':
        manager.print_status()
        
    elif args.command == 'monitor':
        try:
            while True:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print(f"=== Worker Monitor (updates every {args.interval}s) ===")
                print(f"Server: {args.server}:{args.worker_port}")
                print(f"Press Ctrl+C to exit")
                
                manager.print_status()
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\nMonitor stopped")
    else:
        print(f"Unknown command: {args.command}")
        print("Valid commands: start, stop, status, monitor")


if __name__ == "__main__":
    main()