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
            
            # Connect to server (client port is worker_port - 1)
            client_port = self.worker_port - 1
            socket.connect(f"tcp://{self.server_address}:{client_port}")
            
            # Generate a unique client ID - use system hostname
            try:
                hostname = socket.gethostname()
            except:
                hostname = "unknown-host"
                
            client_id = f"worker_manager-{os.getpid()}"
            
            # Send worker status request
            request = {
                'type': 'get_worker_status',
                'client_id': client_id
            }
            
            # Convert to JSON and send
            message = json.dumps(request).encode('utf-8')
            socket.send(message)
            
            # Wait for response with timeout
            if socket.poll(timeout=5000):  # 5 second timeout
                response_data = socket.recv()
                response = json.loads(response_data.decode('utf-8'))
                
                socket.close()
                context.term()
                
                return response
            
            socket.close()
            context.term()
            
            logger.warning("Request for worker status timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error getting worker status from server: {str(e)}")
            return None
    
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
        
        status = self.get_worker_status_from_server()
        
        if status and status.get('status') == 'success':
            workers = status.get('workers', [])
            
            if not workers:
                print("No workers connected to server")
            else:
                # Create table
                table_data = []
                
                for worker in workers:
                    node_id = worker.get('node_id', '')
                    address = worker.get('address', '')
                    status = worker.get('status', '')
                    
                    # Get stats
                    stats = worker.get('stats', {})
                    completed_tasks = stats.get('completed_tasks', 0)
                    avg_time = stats.get('avg_processing_time', 0)
                    
                    # Format average time
                    avg_time_str = f"{avg_time:.2f}s" if avg_time else "N/A"
                    
                    # Get last heartbeat time
                    last_heartbeat = worker.get('last_heartbeat', 0)
                    time_since_heartbeat = time.time() - last_heartbeat
                    
                    table_data.append([
                        node_id, 
                        address, 
                        status, 
                        completed_tasks,
                        avg_time_str,
                        f"{time_since_heartbeat:.1f}s"
                    ])
                
                # Print table
                print(tabulate(table_data, headers=[
                    "Node ID", "Address", "Status", "Tasks", "Avg Time", "Last Heartbeat"
                ]))
                
                # Print summary
                print(f"\nTotal workers: {status.get('worker_count', 0)}")
                print(f"Idle workers: {status.get('idle_workers', 0)}")
                print(f"Pending tasks: {status.get('pending_tasks', 0)}")
                
        else:
            print("Could not get status from server")
        
        print()


if __name__ == "__main__":
    # Parse command line arguments - simplified version
    parser = argparse.ArgumentParser(description='Distributed Word Count Worker Manager')
    
    # Common parameters
    parser.add_argument('--server', default='localhost', help='Server address')
    parser.add_argument('--worker-port', type=int, default=5556, help='Server worker port')
    parser.add_argument('--heartbeat-port', type=int, default=5557, help='Server heartbeat port')
    parser.add_argument('--worker-script', default='worker.py', help='Path to worker script')
    
    # Command and count
    parser.add_argument('command', nargs='?', default='status', help='Command: start, stop, status, monitor')
    parser.add_argument('count', nargs='?', type=int, default=1, help='Number of workers (for start command)')
    parser.add_argument('--interval', type=int, default=5, help='Update interval (for monitor command)')
    
    args = parser.parse_args()
    
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