# client.py
import os
import time
import zmq
import json
import threading
import uuid
import base64
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import argparse
import socket
import re

# Import the existing GUI components
from word_cloud import WordCloudTab
from query_processor import QueryTab, InvertedIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WordCountClient')

class DistributedClient:
    """Client that communicates with the distributed server."""
    def __init__(self, server_address, server_port=5555):
        self.server_address = server_address
        self.server_port = server_port
        
        # Generate client ID
        self.client_id = f"client-{uuid.uuid4().hex[:8]}"
        
        # ZeroMQ context and socket
        self.context = zmq.Context()
        self.server_socket = self.context.socket(zmq.REQ)
        
        # Connect to server
        server_connection = f"tcp://{server_address}:{server_port}"
        self.server_socket.connect(server_connection)
        logger.info(f"Connected to server at {server_connection}")
        
        # Request timeout
        self.request_timeout = 30000  # 30 seconds in ms
    
    def close(self):
        """Close connection to server."""
        self.server_socket.close()
        self.context.term()
    
    def process_files(self, filepaths):
        """
        Send files to server for processing.
        
        Args:
            filepaths: List of file paths to process
            
        Returns:
            Response from server
        """
        file_data = []
        
        # Read and encode file content
        for filepath in filepaths:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    file_data.append({
                        'path': os.path.basename(filepath),
                        'content': content
                    })
            except Exception as e:
                logger.error(f"Error reading file {filepath}: {str(e)}")
        
        request = {
            'type': 'process_files',
            'client_id': self.client_id,
            'file_data': file_data
        }
        
        # Send request to server
        self.server_socket.send(json.dumps(request).encode('utf-8'))
        
        # Wait for response with timeout
        if self.server_socket.poll(timeout=self.request_timeout):
            response_data = self.server_socket.recv()
            response = json.loads(response_data.decode('utf-8'))
            return response
        else:
            logger.error("Request timed out")
            return {
                'status': 'error',
                'message': 'Request timed out'
            }
    
    def get_results(self):
        """
        Get processing results from server.
        
        Returns:
            Processing results
        """
        request = {
            'type': 'get_results',
            'client_id': self.client_id
        }
        
        # Send request to server
        self.server_socket.send(json.dumps(request).encode('utf-8'))
        
        # Wait for response with timeout
        if self.server_socket.poll(timeout=self.request_timeout):
            response_data = self.server_socket.recv()
            response = json.loads(response_data.decode('utf-8'))
            return response
        else:
            logger.error("Request timed out")
            return {
                'status': 'error',
                'message': 'Request timed out'
            }
    
    def get_worker_status(self):
        """
        Get status of worker nodes.
        
        Returns:
            Worker status information
        """
        request = {
            'type': 'get_worker_status',
            'client_id': self.client_id
        }
        
        # Send request to server
        self.server_socket.send(json.dumps(request).encode('utf-8'))
        
        # Wait for response with timeout
        if self.server_socket.poll(timeout=self.request_timeout):
            response_data = self.server_socket.recv()
            response = json.loads(response_data.decode('utf-8'))
            return response
        else:
            logger.error("Request timed out")
            return {
                'status': 'error',
                'message': 'Request timed out'
            }


class DistributedFileProcessor:
    """
    A mock version of ImprovedFileProcessor that works with the distributed system.
    This class provides compatibility with the original tabs.
    """
    def __init__(self):
        self.results = []  # [(chunk_id, word_counts)]
        self.file_chunks = []  # [(chunk_text, chunk_id)]
        self.file_map = {}  # chunk_id -> file_path
        self.total_word_counts = Counter()
    
    def update_from_server_results(self, server_results, filepaths):
        """Update internal state from server results."""
        word_counts = server_results.get('all_counts', {})
        
        # Convert to Counter for compatibility
        self.total_word_counts = Counter(word_counts)
        
        # Reset results and create a synthetic result for compatibility
        self.results = [(0, self.total_word_counts)]
        
        # Create file chunks for the search tab
        self.file_chunks = []
        
        # Map filepaths to document IDs
        for i, filepath in enumerate(filepaths):
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    # Create chunks from file (simplified)
                    chunks = self._split_content(content)
                    
                    for j, chunk_text in enumerate(chunks):
                        chunk_id = i * 1000 + j  # Ensure unique chunk IDs
                        self.file_chunks.append((chunk_text, chunk_id))
                        self.file_map[chunk_id] = filepath
                        
            except Exception as e:
                logger.error(f"Error reading file {filepath}: {str(e)}")
    
    def _split_content(self, content, chunk_size=100000):
        """Split content into chunks."""
        if not content:
            return []
        
        # Simple splitting by size
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i+chunk_size])
        
        return chunks


class DistributedClientGUI:
    """
    GUI application for the distributed word count system.
    This adapts the existing GUI to work with the distributed backend.
    """
    def __init__(self, root, server_address="localhost", server_port=5555):
        self.root = root
        self.root.title("Distributed Word Count System")
        self.root.geometry("1000x800")
        
        # Create distributed client
        self.client = DistributedClient(server_address, server_port)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create main tab
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Word Count")
        
        # File selection
        self.frame_file = ttk.Frame(self.main_tab, padding="10")
        self.frame_file.pack(fill=tk.X)
        
        ttk.Label(self.frame_file, text="Select text files:").pack(side=tk.LEFT)
        self.selected_files_var = tk.StringVar()
        self.selected_files_var.set("No files selected")
        ttk.Label(self.frame_file, textvariable=self.selected_files_var).pack(side=tk.LEFT, padx=10)
        ttk.Button(self.frame_file, text="Browse...", command=self.select_files).pack(side=tk.RIGHT)
        
        # Server connection
        self.frame_connection = ttk.Frame(self.main_tab, padding="10")
        self.frame_connection.pack(fill=tk.X)
        
        ttk.Label(self.frame_connection, text="Server:").pack(side=tk.LEFT)
        self.server_var = tk.StringVar(value=server_address)
        ttk.Entry(self.frame_connection, textvariable=self.server_var, width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.frame_connection, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_var = tk.StringVar(value=str(server_port))
        ttk.Entry(self.frame_connection, textvariable=self.port_var, width=6).pack(side=tk.LEFT)
        
        ttk.Button(self.frame_connection, text="Connect", command=self.connect_to_server).pack(side=tk.LEFT, padx=10)
        
        # Processing options
        self.frame_options = ttk.Frame(self.main_tab, padding="10")
        self.frame_options.pack(fill=tk.X)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(self.frame_options, orient="horizontal", variable=self.progress_var)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(self.frame_options, text="Process Files", command=self.process_files).pack(side=tk.RIGHT)
        
        # Results area
        self.frame_results = ttk.LabelFrame(self.main_tab, text="Processing Results", padding="10")
        self.frame_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(self.frame_results, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization frame
        self.frame_viz = ttk.LabelFrame(self.main_tab, text="Word Frequency Visualization", padding="10")
        self.frame_viz.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Worker status tab
        self.worker_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.worker_tab, text="Worker Status")
        self._setup_worker_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Connected to server at {server_address}:{server_port}")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Selected files
        self.selected_files = []
        
        # Store the last results
        self.last_word_counts = None
        
        # Create mock processor for compatibility with existing tabs
        self.processor = DistributedFileProcessor()
        
        # Add Word Cloud tab
        self.wordcloud_tab = WordCloudTab(self.notebook, self.processor)
        
        # Add Search tab
        self.search_tab = QueryTab(self.notebook, self.processor)
        
        # Set up polling for results
        self.polling = False
        self.poll_thread = None
    
    def _setup_worker_tab(self):
        """Set up the worker status tab."""
        # Main frame
        worker_frame = ttk.Frame(self.worker_tab, padding="10")
        worker_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls
        control_frame = ttk.Frame(worker_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Refresh Worker Status", command=self.update_worker_status).pack(side=tk.LEFT)
        
        self.auto_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Auto-refresh", variable=self.auto_refresh_var, 
                        command=self.toggle_auto_refresh).pack(side=tk.LEFT, padx=10)
        
        # Worker list
        list_frame = ttk.LabelFrame(worker_frame, text="Worker Nodes", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for worker list
        columns = ("node_id", "address", "status", "tasks", "avg_time")
        self.worker_tree = ttk.Treeview(list_frame, columns=columns, show="headings")
        
        # Set column headings
        self.worker_tree.heading("node_id", text="Node ID")
        self.worker_tree.heading("address", text="Address")
        self.worker_tree.heading("status", text="Status")
        self.worker_tree.heading("tasks", text="Completed Tasks")
        self.worker_tree.heading("avg_time", text="Avg Time (s)")
        
        # Set column widths
        self.worker_tree.column("node_id", width=150)
        self.worker_tree.column("address", width=120)
        self.worker_tree.column("status", width=80)
        self.worker_tree.column("tasks", width=100)
        self.worker_tree.column("avg_time", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.worker_tree.yview)
        self.worker_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.worker_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Stats frame
        stats_frame = ttk.LabelFrame(worker_frame, text="System Stats", padding="10")
        stats_frame.pack(fill=tk.X, pady=5)
        
        # Stats labels
        self.total_workers_var = tk.StringVar(value="Total Workers: 0")
        self.idle_workers_var = tk.StringVar(value="Idle Workers: 0")
        self.pending_tasks_var = tk.StringVar(value="Pending Tasks: 0")
        
        ttk.Label(stats_frame, textvariable=self.total_workers_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(stats_frame, textvariable=self.idle_workers_var).pack(side=tk.LEFT, padx=10)
        ttk.Label(stats_frame, textvariable=self.pending_tasks_var).pack(side=tk.LEFT, padx=10)
    
    def connect_to_server(self):
        """Connect to the specified server."""
        server = self.server_var.get()
        
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Invalid Port", "Please enter a valid port number")
            return
        
        try:
            # Close existing client if any
            if hasattr(self, 'client'):
                self.client.close()
            
            # Create new client
            self.client = DistributedClient(server, port)
            
            # Test connection by getting worker status
            response = self.client.get_worker_status()
            
            if response.get('status') == 'success':
                self.status_var.set(f"Connected to server at {server}:{port}")
                messagebox.showinfo("Connection Successful", 
                                    f"Connected to server with {response.get('worker_count', 0)} workers")
                
                # Update worker status
                self.update_worker_status()
            else:
                self.status_var.set(f"Error connecting to server: {response.get('message', 'Unknown error')}")
                messagebox.showerror("Connection Error", response.get('message', 'Unknown error'))
                
        except Exception as e:
            self.status_var.set(f"Error connecting to server: {str(e)}")
            messagebox.showerror("Connection Error", str(e))
    
    def select_files(self):
        """Open file dialog to select text files."""
        filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        self.selected_files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if self.selected_files:
            self.selected_files_var.set(f"{len(self.selected_files)} files selected")
        else:
            self.selected_files_var.set("No files selected")
    
    def process_files(self):
        """Process the selected files using the distributed system."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        # Update status
        self.status_var.set("Sending files to server...")
        self.progress_var.set(0)
        self.root.update()
        
        # Send files to server
        response = self.client.process_files(self.selected_files)
        
        if response.get('status') != 'success':
            self.status_var.set(f"Error: {response.get('message', 'Unknown error')}")
            messagebox.showerror("Processing Error", response.get('message', 'Unknown error'))
            return
        
        # Start polling for results
        self.status_var.set(response.get('message', 'Processing started'))
        self.start_polling()
    
    def start_polling(self):
        """Start polling for results."""
        if self.polling:
            return
        
        self.polling = True
        
        # Start polling thread
        self.poll_thread = threading.Thread(target=self._poll_results)
        self.poll_thread.daemon = True
        self.poll_thread.start()
    
    def stop_polling(self):
        """Stop polling for results."""
        self.polling = False
        
        if self.poll_thread and self.poll_thread.is_alive():
            self.poll_thread.join(timeout=1.0)
    
    def _poll_results(self):
        """Poll for processing results."""
        poll_interval = 1.0  # seconds
        
        while self.polling:
            try:
                # Get results from server
                response = self.client.get_results()
                
                if response.get('status') == 'in_progress':
                    # Update progress
                    remaining = response.get('remaining', 0)
                    total = response.get('total', 1)
                    
                    if total > 0:
                        progress = 100 * (total - remaining) / total
                        
                        # Update UI from main thread
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                        self.root.after(0, lambda m=response.get('message', 'Processing in progress'): 
                                        self.status_var.set(m))
                    
                elif response.get('status') == 'complete':
                    # Processing complete
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.status_var.set("Processing complete"))
                    
                    # Process results
                    results = response.get('results', {})
                    self.root.after(0, lambda r=results: self._display_results(r))
                    
                    # Stop polling
                    self.polling = False
                    break
                    
                elif response.get('status') == 'error':
                    # Error occurred
                    self.root.after(0, lambda m=response.get('message', 'Error'): 
                                    self.status_var.set(f"Error: {m}"))
                    self.root.after(0, lambda m=response.get('message', 'Error'): 
                                    messagebox.showerror("Processing Error", m))
                    
                    # Stop polling
                    self.polling = False
                    break
            
            except Exception as e:
                logger.error(f"Error polling for results: {str(e)}")
                self.root.after(0, lambda e=str(e): self.status_var.set(f"Error: {e}"))
                
                # Stop polling on error
                self.polling = False
                break
            
            # Sleep before next poll
            time.sleep(poll_interval)
    
    def _display_results(self, results):
        """Display processing results."""
        if not results:
            return
        
        # Extract counts
        word_counts = results.get('all_counts', {})
        total_words = results.get('total_words', 0)
        unique_words = results.get('unique_words', 0)
        
        # Convert to Counter for compatibility with other tabs
        self.last_word_counts = Counter(word_counts)
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Processing completed\n\n")
        self.results_text.insert(tk.END, f"Total words: {total_words}\n")
        self.results_text.insert(tk.END, f"Total unique words: {unique_words}\n\n")
        self.results_text.insert(tk.END, "Top 20 most common words:\n")
        
        # Display top words
        for word, count in Counter(word_counts).most_common(20):
            self.results_text.insert(tk.END, f"{word}: {count}\n")
        
        # Visualize results
        self.visualize_results(Counter(word_counts))
        
        # Update other tabs
        self._update_other_tabs(Counter(word_counts))
    
    def visualize_results(self, word_counts):
        """Create visualization of word count results."""
        # Clear previous visualization
        for widget in self.frame_viz.winfo_children():
            widget.destroy()
        
        # Create figure for visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Get top 10 words for visualization
        top_words = dict(word_counts.most_common(10))
        words = list(top_words.keys())
        counts = list(top_words.values())
        
        # Create horizontal bar chart
        bars = ax.barh(words, counts, color='skyblue')
        ax.set_title('Top 10 Most Common Words')
        ax.set_xlabel('Count')
        
        # Add count labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width}', 
                    ha='left', va='center')
        
        # Add the plot to tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.frame_viz)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _update_other_tabs(self, word_counts):
        """Update other tabs with results."""
        # Update processor with results
        self.processor.update_from_server_results(
            {'all_counts': dict(word_counts)}, 
            self.selected_files
        )
        
        # Update word cloud tab
        self.wordcloud_tab.update_word_counts(word_counts)
        
        # Update search tab with new processor
        self.search_tab.update_processor(self.processor, self.selected_files)
    
    def update_worker_status(self):
        """Update the worker status display."""
        try:
            response = self.client.get_worker_status()
            
            if response.get('status') == 'success':
                # Clear existing items
                for item in self.worker_tree.get_children():
                    self.worker_tree.delete(item)
                
                # Add workers to tree
                workers = response.get('workers', [])
                for worker in workers:
                    node_id = worker.get('node_id', '')
                    address = worker.get('address', '')
                    status = worker.get('status', '')
                    
                    # Get stats
                    stats = worker.get('stats', {})
                    completed_tasks = stats.get('completed_tasks', 0)
                    avg_time = stats.get('avg_processing_time', 0)
                    
                    # Format average time
                    avg_time_str = f"{avg_time:.2f}" if avg_time else "N/A"
                    
                    # Add to tree
                    self.worker_tree.insert("", "end", values=(
                        node_id, address, status, completed_tasks, avg_time_str
                    ))
                
                # Update stats
                self.total_workers_var.set(f"Total Workers: {response.get('worker_count', 0)}")
                self.idle_workers_var.set(f"Idle Workers: {response.get('idle_workers', 0)}")
                self.pending_tasks_var.set(f"Pending Tasks: {response.get('pending_tasks', 0)}")
                
                self.status_var.set("Worker status updated")
            else:
                self.status_var.set(f"Error updating worker status: {response.get('message', 'Unknown error')}")
        except Exception as e:
            self.status_var.set(f"Error updating worker status: {str(e)}")
    
    def toggle_auto_refresh(self):
        """Toggle automatic refresh of worker status."""
        if self.auto_refresh_var.get():
            # Start auto-refresh
            self.auto_refresh()
        # If turned off, it will just stop the next cycle
    
    def auto_refresh(self):
        """Automatically refresh worker status."""
        if self.auto_refresh_var.get():
            self.update_worker_status()
            # Schedule next refresh after 5 seconds
            self.root.after(5000, self.auto_refresh)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Distributed Word Count Client')
    parser.add_argument('--server', default='localhost', help='Server address')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger('WordCountClient').setLevel(logging.DEBUG)
    
    # Create GUI
    root = tk.Tk()
    app = DistributedClientGUI(root, args.server, args.port)
    root.mainloop()