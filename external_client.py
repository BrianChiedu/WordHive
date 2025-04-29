#!/usr/bin/env python3
"""
WordHive Client - Connect to a WordHive server to process text files

This simple client allows students to connect to a centrally hosted WordHive server
to process text files and visualize the results.
"""
import os
import sys
import time
import json
import zmq
import argparse
import threading
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

class WordHiveClient:
    """Client for connecting to a remote WordHive server."""
    
    def __init__(self, server_address, server_port=5555):
        """
        Initialize the client.
        
        Args:
            server_address: Address of the WordHive server
            server_port: Port of the WordHive server
        """
        self.server_address = server_address
        self.server_port = server_port
        self.client_id = f"student-{os.getenv('USER', 'user')}-{int(time.time())}"
        
        # ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = None
        self.connected = False
        
    def connect(self):
        """Connect to the server."""
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout
            
            server_url = f"tcp://{self.server_address}:{self.server_port}"
            print(f"Connecting to WordHive server at {server_url}...")
            self.socket.connect(server_url)
            
            # Test connection by getting worker status
            self.connected = True
            worker_status = self.get_worker_status()
            
            if worker_status and worker_status.get('status') == 'success':
                worker_count = worker_status.get('worker_count', 0)
                idle_workers = worker_status.get('idle_workers', 0)
                print(f"Connected to server with {worker_count} workers ({idle_workers} idle)")
                return True
            else:
                print("Connection failed: Server not responding properly")
                self.disconnect()
                return False
                
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            if self.socket:
                self.socket.close()
                self.socket = None
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the server."""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
    
    def process_files(self, file_paths):
        """
        Send files to the server for processing.
        
        Args:
            file_paths: List of paths to text files
            
        Returns:
            Dictionary with processing results or None on error
        """
        if not self.connected:
            print("Not connected to server")
            return None
        
        if not file_paths:
            print("No files specified")
            return None
        
        # Read file contents
        file_data = []
        
        print("Reading files...")
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                file_data.append({
                    'path': os.path.basename(file_path),
                    'content': content
                })
                print(f"Read {file_path}: {len(content)} characters")
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
        
        if not file_data:
            print("No valid files to process")
            return None
        
        # Prepare request
        request = {
            'type': 'process_files',
            'client_id': self.client_id,
            'file_data': file_data
        }
        
        try:
            # Send request
            print(f"Sending {len(file_data)} files to server...")
            self.socket.send(json.dumps(request).encode('utf-8'))
            
            # Get response
            response_data = self.socket.recv()
            response = json.loads(response_data.decode('utf-8'))
            
            if response.get('status') == 'success':
                print(response.get('message', 'Processing started'))
                task_count = response.get('task_count', 0)
                
                # Poll for results
                results = self._poll_for_results(task_count)
                return results
            else:
                print(f"Error: {response.get('message', 'Unknown error')}")
                return None
                
        except zmq.error.Again:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Error processing files: {str(e)}")
            return None
    
    def _poll_for_results(self, task_count):
        """
        Poll the server for processing results.
        
        Args:
            task_count: Number of tasks to wait for
            
        Returns:
            Processing results
        """
        print("Waiting for processing to complete...")
        
        # Create progress bar
        progress_bar = tqdm(total=task_count, desc="Processing")
        
        # Poll until complete
        last_remaining = task_count
        
        while True:
            try:
                # Prepare request
                request = {
                    'type': 'get_results',
                    'client_id': self.client_id
                }
                
                # Send request
                self.socket.send(json.dumps(request).encode('utf-8'))
                
                # Get response
                response_data = self.socket.recv()
                response = json.loads(response_data.decode('utf-8'))
                
                if response.get('status') == 'in_progress':
                    # Update progress bar
                    remaining = response.get('remaining', 0)
                    if remaining < last_remaining:
                        progress_bar.update(last_remaining - remaining)
                        last_remaining = remaining
                    
                    # Brief pause before next poll
                    time.sleep(0.5)
                    
                elif response.get('status') == 'complete':
                    # Complete progress bar
                    progress_bar.update(last_remaining)
                    progress_bar.close()
                    
                    print("Processing complete!")
                    return response.get('results', {})
                    
                else:
                    progress_bar.close()
                    print(f"Error: {response.get('message', 'Unknown error')}")
                    return None
                    
            except zmq.error.Again:
                progress_bar.close()
                print("Polling timed out")
                return None
            except Exception as e:
                progress_bar.close()
                print(f"Error polling for results: {str(e)}")
                return None
    
    def get_worker_status(self):
        """
        Get status of worker nodes.
        
        Returns:
            Worker status information
        """
        if not self.connected:
            print("Not connected to server")
            return None
        
        try:
            # Prepare request
            request = {
                'type': 'get_worker_status',
                'client_id': self.client_id
            }
            
            # Send request
            self.socket.send(json.dumps(request).encode('utf-8'))
            
            # Get response
            response_data = self.socket.recv()
            response = json.loads(response_data.decode('utf-8'))
            
            return response
                
        except zmq.error.Again:
            print("Request timed out")
            return None
        except Exception as e:
            print(f"Error getting worker status: {str(e)}")
            return None
    
    def visualize_results(self, results):
        """
        Visualize word count results.
        
        Args:
            results: Processing results from the server
        """
        if not results:
            print("No results to visualize")
            return
        
        # Extract counts
        word_counts = results.get('all_counts', {})
        total_words = results.get('total_words', 0)
        unique_words = results.get('unique_words', 0)
        
        print(f"\nResults Summary:")
        print(f"Total words: {total_words}")
        print(f"Unique words: {unique_words}")
        
        # Display top words
        top_words = Counter(word_counts).most_common(20)
        
        print("\nTop 20 most common words:")
        for word, count in top_words:
            print(f"{word}: {count}")
        
        # Create visualization
        self._create_word_chart(top_words)
    
    def _create_word_chart(self, top_words):
        """
        Create a bar chart of the top words.
        
        Args:
            top_words: List of (word, count) tuples
        """
        # Extract words and counts
        words = [word for word, _ in top_words[:10]]
        counts = [count for _, count in top_words[:10]]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.barh(words, counts, color='skyblue')
        plt.xlabel('Count')
        plt.title('Top 10 Most Common Words')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add count labels to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width}', 
                     ha='left', va='center')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='WordHive Client')
    parser.add_argument('--server', required=True, help='Server address')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('files', nargs='*', help='Text files to process')
    args = parser.parse_args()
    
    # Create client
    client = WordHiveClient(args.server, args.port)
    
    # Connect to server
    if not client.connect():
        return 1
    
    try:
        # Process files if provided
        if args.files:
            results = client.process_files(args.files)
            if results:
                client.visualize_results(results)
        else:
            print("No files specified. Use the --help option for usage information.")
            print("Example: python wordhive_client.py --server classroom.example.edu file1.txt file2.txt")
    finally:
        # Disconnect from server
        client.disconnect()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())