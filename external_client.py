#!/usr/bin/env python3
"""
WordHive External Client - Connect to a WordHive server to process text files

This client provides students with a complete GUI interface to the WordHive server,
allowing them to process text files, visualize results, search content, and more.
"""
import os
import sys
import time
import json
import zmq
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter, defaultdict
import logging
import argparse
import socket
import re
import math
import heapq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('WordHiveExternalClient')

class InvertedIndex:
    """
    Creates and manages an inverted index to enable document search.
    An inverted index maps each word to the documents it appears in.
    """
    def __init__(self):
        self.index = defaultdict(list)  # word -> [(doc_id, frequency, positions)]
        self.doc_map = {}  # doc_id -> document path
        self.doc_lengths = {}  # doc_id -> total words in document
        self.total_docs = 0
        self.doc_chunks = {}  # doc_id -> list of chunk texts (for displaying snippets)
    
    def add_document(self, doc_id, filepath, chunks):
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique identifier for the document
            filepath: Path to the document file
            chunks: List of text chunks from the document
        """
        self.doc_map[doc_id] = filepath
        self.doc_chunks[doc_id] = chunks
        self.total_docs += 1
    
    def add_chunk_results(self, doc_id, chunk_id, word_counts, chunk_text):
        """
        Add word counts from a text chunk to the index.
        
        Args:
            doc_id: Document identifier
            chunk_id: Chunk identifier within the document
            word_counts: Counter of words and their frequencies
            chunk_text: The text of the chunk (for displaying snippets)
        """
        # Store chunk text
        if doc_id not in self.doc_chunks:
            self.doc_chunks[doc_id] = {}
        self.doc_chunks[doc_id][chunk_id] = chunk_text
        
        # Extract word positions from chunk
        word_positions = {}
        position = 0
        for match in re.finditer(r'\b[a-z][a-z0-9_]*\b', chunk_text.lower()):
            word = match.group(0)
            if word not in word_positions:
                word_positions[word] = []
            word_positions[word].append(position)
            position += 1
        
        # Update index with word frequencies and positions
        for word, count in word_counts.items():
            # Create entry for this word if it doesn't exist for this document
            if not any(entry[0] == doc_id for entry in self.index[word]):
                self.index[word].append([doc_id, count, word_positions.get(word, [])])
            else:
                # Update existing entry
                for entry in self.index[word]:
                    if entry[0] == doc_id:
                        entry[1] += count
                        entry[2].extend(word_positions.get(word, []))
                        break
        
        # Update document length
        if doc_id not in self.doc_lengths:
            self.doc_lengths[doc_id] = 0
        self.doc_lengths[doc_id] += sum(word_counts.values())
    
    def prepare_index(self):
        """Prepare the index for searching by sorting and computing statistics."""
        # Sort positions for each word in each document
        for word in self.index:
            for entry in self.index[word]:
                entry[2].sort()
    
    def search(self, query, max_results=10):
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of (doc_id, score, snippet) tuples for matching documents
        """
        # Clean and tokenize the query
        query_words = self._tokenize_query(query)
        
        if not query_words:
            return []
        
        # Compute TF-IDF scores for documents containing any query word
        scores = defaultdict(float)
        
        for word in query_words:
            if word in self.index:
                # Calculate IDF (Inverse Document Frequency)
                idf = math.log(self.total_docs / len(self.index[word]))
                
                # For each document containing this word
                for doc_id, freq, positions in self.index[word]:
                    # Calculate TF (Term Frequency)
                    if self.doc_lengths[doc_id] > 0:
                        tf = freq / self.doc_lengths[doc_id]
                        scores[doc_id] += tf * idf
        
        # Get top matching documents
        top_docs = heapq.nlargest(max_results, scores.items(), key=lambda x: x[1])
        
        results = []
        for doc_id, score in top_docs:
            # Generate snippet for this document
            snippet = self._generate_snippet(doc_id, query_words)
            
            # Get document filename (basename of path)
            filename = os.path.basename(self.doc_map[doc_id])
            
            results.append((doc_id, filename, score, snippet))
        
        return results
    
    def _tokenize_query(self, query):
        """Convert query string to list of cleaned tokens."""
        # Convert to lowercase and remove punctuation
        cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Split by whitespace and filter to words
        tokens = [word for word in cleaned_query.split() 
                 if re.match(r'^[a-z][a-z0-9_]*$', word) and len(word) >= 2]
        
        return tokens
    
    def _generate_snippet(self, doc_id, query_words, context_size=40):
        """Generate a snippet of text containing query words for a document."""
        # Find the chunk with the most query words
        best_chunk_id = None
        best_chunk_score = -1
        
        if doc_id not in self.doc_chunks:
            return "No snippet available"
        
        for chunk_id, chunk_text in self.doc_chunks[doc_id].items():
            chunk_lower = chunk_text.lower()
            score = sum(chunk_lower.count(word) for word in query_words)
            if score > best_chunk_score:
                best_chunk_score = score
                best_chunk_id = chunk_id
        
        if best_chunk_id is None:
            return "No matching content found"
        
        # Get the best chunk text
        chunk_text = self.doc_chunks[doc_id][best_chunk_id]
        chunk_lower = chunk_text.lower()
        
        # Find the best position for the snippet
        best_pos = 0
        best_pos_score = -1
        
        for i in range(len(chunk_lower)):
            score = sum(1 for word in query_words if chunk_lower[i:i+len(word)] == word)
            if score > best_pos_score:
                best_pos_score = score
                best_pos = i
        
        # Extract snippet centered around the best position
        start = max(0, best_pos - context_size)
        end = min(len(chunk_text), best_pos + context_size)
        
        snippet = chunk_text[start:end]
        
        # Add ellipsis if necessary
        if start > 0:
            snippet = "..." + snippet
        if end < len(chunk_text):
            snippet = snippet + "..."
        
        return snippet
    
    def get_document_path(self, doc_id):
        """Get the filepath for a document ID."""
        return self.doc_map.get(doc_id)
    
    def get_document_text(self, doc_id):
        """Get the full text for a document ID by concatenating chunks."""
        if doc_id not in self.doc_chunks:
            return "Document not found"
        
        chunks = self.doc_chunks[doc_id]
        if isinstance(chunks, dict):
            # If chunks are stored by chunk_id
            return " ".join(chunks.values())
        else:
            # If chunks are stored as a list
            return " ".join(chunks)

class QueryProcessor:
    """
    Provides search functionality for indexed documents.
    """
    def __init__(self):
        self.inverted_index = InvertedIndex()
        self.is_indexing = False
    
    def index_documents(self, filepaths, content_map=None):
        """
        Build inverted index from documents.
        
        Args:
            filepaths: List of file paths to index
            content_map: Optional map of filepath -> content to avoid rereading files
        """
        self.is_indexing = True
        
        # Map file paths to document IDs
        for doc_id, filepath in enumerate(filepaths):
            # Read content if not provided
            if content_map and filepath in content_map:
                content = content_map[filepath]
            else:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {str(e)}")
                    continue
            
            # Split content into chunks
            chunks = self._split_content(content)
            
            # Add document to index
            chunk_dict = {i: chunk for i, chunk in enumerate(chunks)}
            self.inverted_index.add_document(doc_id, filepath, chunk_dict)
            
            # Process content
            for chunk_id, chunk_text in enumerate(chunks):
                # Count words in chunk
                word_counts = self._count_words(chunk_text)
                
                # Add to index
                self.inverted_index.add_chunk_results(doc_id, chunk_id, word_counts, chunk_text)
        
        # Prepare index for searching
        self.inverted_index.prepare_index()
        
        self.is_indexing = False
        return len(filepaths)
    
    def _split_content(self, content, chunk_size=10000):
        """Split content into chunks."""
        if not content:
            return []
        
        # Split by paragraphs first, then combine into chunks
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # If no chunks were created (e.g., content is smaller than chunk_size)
        if not chunks and content:
            chunks = [content]
        
        return chunks
    
    def _count_words(self, text):
        """Count word frequencies in text."""
        if not text:
            return Counter()
        
        # Remove punctuation and normalize whitespace
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split by whitespace
        words = cleaned_text.split()
        
        # Filter to ensure only actual words are counted (at least 2 characters)
        valid_words = [word for word in words 
                       if re.match(r'^[a-z][a-z0-9_]*$', word) and len(word) >= 2]
        
        # Count words
        return Counter(valid_words)
    
    def search(self, query, max_results=10):
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not query or self.is_indexing:
            return []
        
        return self.inverted_index.search(query, max_results)
    
    def get_document(self, doc_id):
        """
        Get the full text of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document text
        """
        return self.inverted_index.get_document_text(doc_id)

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
        
        # Store processed files and their content
        self.processed_files = []
        self.file_content_map = {}
        
        # Search functionality
        self.query_processor = QueryProcessor()
        
    def connect(self):
        """Connect to the server."""
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, 10000)  # 10 second timeout
            
            server_url = f"tcp://{self.server_address}:{self.server_port}"
            logger.info(f"Connecting to WordHive server at {server_url}...")
            self.socket.connect(server_url)
            
            # Test connection by getting worker status
            self.connected = True
            worker_status = self.get_worker_status()
            
            if worker_status and worker_status.get('status') == 'success':
                worker_count = worker_status.get('worker_count', 0)
                idle_workers = worker_status.get('idle_workers', 0)
                logger.info(f"Connected to server with {worker_count} workers ({idle_workers} idle)")
                return True
            else:
                logger.error("Connection failed: Server not responding properly")
                self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
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
            logger.error("Not connected to server")
            return None
        
        if not file_paths:
            logger.error("No files specified")
            return None
        
        # Clear previous processed files
        self.processed_files = []
        self.file_content_map = {}
        
        # Read file contents
        file_data = []
        
        logger.info("Reading files...")
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                file_data.append({
                    'path': os.path.basename(file_path),
                    'content': content
                })
                logger.info(f"Read {file_path}: {len(content)} characters")
                
                # Store file content for search indexing
                self.file_content_map[file_path] = content
                self.processed_files.append(file_path)
                
            except Exception as e:
                logger.error(f"Error reading {file_path}: {str(e)}")
        
        if not file_data:
            logger.error("No valid files to process")
            return None
        
        # Prepare request
        request = {
            'type': 'process_files',
            'client_id': self.client_id,
            'file_data': file_data
        }
        
        try:
            # Send request
            logger.info(f"Sending {len(file_data)} files to server...")
            self.socket.send(json.dumps(request).encode('utf-8'))
            
            # Get response
            response_data = self.socket.recv()
            response = json.loads(response_data.decode('utf-8'))
            
            if response.get('status') == 'success':
                logger.info(response.get('message', 'Processing started'))
                task_count = response.get('task_count', 0)
                
                # Poll for results
                results = self._poll_for_results(task_count)
                
                # Build search index in background
                if results:
                    threading.Thread(target=self._build_search_index, daemon=True).start()
                
                return results
            else:
                logger.error(f"Error: {response.get('message', 'Unknown error')}")
                return None
                
        except zmq.error.Again:
            logger.error("Request timed out")
            return None
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")
            return None
    
    def _poll_for_results(self, task_count):
        """
        Poll the server for processing results.
        
        Args:
            task_count: Number of tasks to wait for
            
        Returns:
            Processing results
        """
        logger.info("Waiting for processing to complete...")
        
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
                    # Update progress
                    remaining = response.get('remaining', 0)
                    
                    # Brief pause before next poll
                    time.sleep(0.5)
                    
                elif response.get('status') == 'complete':
                    logger.info("Processing complete!")
                    return response.get('results', {})
                    
                else:
                    logger.error(f"Error: {response.get('message', 'Unknown error')}")
                    return None
                    
            except zmq.error.Again:
                logger.error("Polling timed out")
                return None
            except Exception as e:
                logger.error(f"Error polling for results: {str(e)}")
                return None
    
    def get_worker_status(self):
        """
        Get status of worker nodes.
        
        Returns:
            Worker status information
        """
        if not self.connected:
            logger.error("Not connected to server")
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
            logger.error("Request timed out")
            return None
        except Exception as e:
            logger.error(f"Error getting worker status: {str(e)}")
            return None
    
    def _build_search_index(self):
        """Build search index from processed files."""
        if not self.processed_files:
            logger.warning("No files to index")
            return
        
        logger.info("Building search index...")
        try:
            indexed_count = self.query_processor.index_documents(
                self.processed_files, 
                self.file_content_map
            )
            logger.info(f"Indexed {indexed_count} documents for search")
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")

class WordCloudTab:
    """GUI tab for Word Cloud visualization."""
    def __init__(self, parent_notebook, word_processor):
        self.parent_notebook = parent_notebook
        self.word_processor = word_processor
        self.last_word_counts = None
        self.current_figure = None
        
        # Create tab
        self.tab = ttk.Frame(parent_notebook)
        parent_notebook.add(self.tab, text="Word Cloud")
        
        # Main frame with split layout
        main_frame = ttk.Frame(self.tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Word Cloud Settings", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Right panel for visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Word Cloud Visualization", padding="10")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Word cloud display area
        self.wordcloud_frame = ttk.Frame(viz_frame)
        self.wordcloud_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial message
        self.message_var = tk.StringVar(value="Process text files first to generate a word cloud")
        self.message_label = ttk.Label(self.wordcloud_frame, textvariable=self.message_var, font=('TkDefaultFont', 12))
        self.message_label.pack(pady=50)
        
        # Control elements
        ttk.Label(control_frame, text="Max Words:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_words_var = tk.IntVar(value=100)
        ttk.Spinbox(control_frame, from_=20, to=500, increment=10, textvariable=self.max_words_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Background:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.bg_var = tk.StringVar(value="white")
        ttk.Combobox(control_frame, textvariable=self.bg_var, values=["white", "black", "lightgray"], width=10).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Color Map:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.colormap_var = tk.StringVar(value="viridis")
        color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'Blues', 'Greens', 'Reds', 'Purples', 'YlOrBr', 'rainbow']
        ttk.Combobox(control_frame, textvariable=self.colormap_var, values=color_maps, width=10).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(control_frame, text="Min Word Freq:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_freq_var = tk.IntVar(value=3)
        ttk.Spinbox(control_frame, from_=1, to=100, textvariable=self.min_freq_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Generate Cloud", command=self.generate_cloud).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_word_counts(self, word_counts):
        """Update the current word counts."""
        self.last_word_counts = word_counts
        if hasattr(self, 'message_label') and self.message_label.winfo_exists():
            self.message_var.set("Word counts updated. Click 'Generate Cloud' to create visualization.")
    
    def generate_cloud(self):
        """Generate and display word cloud based on current settings."""
        if not self.last_word_counts:
            self.status_var.set("No word count data available. Process files first.")
            return
        
        # Get settings
        max_words = self.max_words_var.get()
        background = self.bg_var.get()
        colormap = self.colormap_var.get()
        min_freq = self.min_freq_var.get()
        
        # Filter words by minimum frequency
        filtered_words = {word: count for word, count in self.last_word_counts.items() 
                         if count >= min_freq}
        
        if not filtered_words:
            self.status_var.set("No words meet the minimum frequency requirement")
            return
        
        # Update status
        self.status_var.set("Generating word cloud...")
        
        # Generate word cloud in a separate thread to avoid UI freezing
        threading.Thread(target=self._generate_cloud_thread, 
                         args=(filtered_words, max_words, background, colormap),
                         daemon=True).start()
    
    def _generate_cloud_thread(self, word_counts, max_words, background, colormap):
        """Thread function to generate word cloud."""
        try:
            # Try to import wordcloud module
            try:
                from wordcloud import WordCloud, STOPWORDS
                have_wordcloud = True
            except ImportError:
                have_wordcloud = False
                
            # Store data for the main thread to process
            self.parent_notebook.after(0, lambda: self._create_wordcloud_in_main_thread(
                word_counts, max_words, background, colormap, have_wordcloud))
                
        except Exception as e:
            error_msg = f"Error generating word cloud: {str(e)}"
            self.parent_notebook.after(0, lambda: self.status_var.set(error_msg))
    
    def _create_wordcloud_in_main_thread(self, word_counts, max_words, background, colormap, have_wordcloud):
        """Create the word cloud visualization in the main thread."""
        try:
            if have_wordcloud:
                # Generate word cloud with wordcloud module
                from wordcloud import WordCloud, STOPWORDS
                wordcloud = WordCloud(
                    background_color=background,
                    max_words=max_words,
                    colormap=colormap,
                    width=800,
                    height=600,
                    stopwords=set(STOPWORDS),
                    min_font_size=10,
                    random_state=42
                ).generate_from_frequencies(word_counts)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                
                # Update display
                self._update_display(fig, len(word_counts))
                
            else:
                # Simple word cloud with matplotlib if wordcloud module is not available
                self._generate_simple_cloud(word_counts, max_words, background, colormap)
                
        except Exception as e:
            error_msg = f"Error generating word cloud: {str(e)}"
            self.status_var.set(error_msg)
    
    def _generate_simple_cloud(self, word_counts, max_words, background, colormap):
        """Generate a simple word frequency plot using matplotlib."""
        # Get top words
        top_words = Counter(word_counts).most_common(max_words)
        words = [w for w, _ in top_words]
        counts = [c for _, c in top_words]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=background)
        
        # Limit to 50 words for readability in bar chart
        display_limit = min(50, len(words))
        
        # Reverse order for better display (largest at top)
        y_pos = range(display_limit)
        bars = ax.barh(y_pos, counts[:display_limit], color=plt.cm.get_cmap(colormap)(range(display_limit)))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words[:display_limit])
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title('Word Frequencies')
        
        # Adjust colors based on background
        text_color = 'black' if background != 'black' else 'white'
        ax.set_xlabel('Frequency', color=text_color)
        ax.tick_params(axis='both', colors=text_color)
        ax.set_facecolor(background)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color(text_color)
        ax.spines['left'].set_color(text_color)
        
        # Update display
        self._update_display(fig, len(word_counts))
    
    def _update_display(self, fig, word_count):
        """Update the UI with the new word cloud figure."""
        # Clear previous display
        for widget in self.wordcloud_frame.winfo_children():
            widget.destroy()
        
        # Store current figure
        self.current_figure = fig
        
        # Display in canvas
        canvas = FigureCanvasTkAgg(fig, master=self.wordcloud_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update status
        self.status_var.set(f"Word cloud generated with {word_count} unique words")
    
    def save_image(self):
        """Save the current word cloud as an image file."""
        if not self.current_figure:
            self.status_var.set("No word cloud to save. Generate one first.")
            return
        
        # Open file dialog
        filetypes = [
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg"),
            ("PDF Document", "*.pdf")
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            title="Save Word Cloud"
        )
        
        if filename:
            try:
                self.current_figure.savefig(filename, bbox_inches='tight')
                self.status_var.set(f"Word cloud saved to {filename}")
            except Exception as e:
                self.status_var.set(f"Error saving file: {str(e)}")

class QueryTab:
    """GUI tab for search functionality."""
    def __init__(self, parent_notebook, client):
        self.parent_notebook = parent_notebook
        self.client = client
        
        # Create tab
        self.tab = ttk.Frame(parent_notebook)
        parent_notebook.add(self.tab, text="Search")
        
        # Create styles
        self._setup_styles()
        self._setup_ui()
    
    def _setup_styles(self):
        """Set up ttk styles needed for this tab."""
        style = ttk.Style()
        
        # Create the result item style if it doesn't exist
        style.configure("ResultItem.TFrame", relief=tk.GROOVE, borderwidth=1, padding=5)
        
        # Create hyperlink style
        style.configure("Hyperlink.TButton", foreground="blue", borderwidth=0, padding=0)
        style.map("Hyperlink.TButton", foreground=[('active', 'purple')])
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main frame
        main_frame = ttk.Frame(self.tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Search bar
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=5)
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=50)
        search_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        search_entry.bind("<Return>", lambda e: self.perform_search())
        
        self.search_button = ttk.Button(search_frame, text="Search", command=self.perform_search)
        self.search_button.pack(side=tk.RIGHT, padx=5)
        
        # Index status
        self.index_status_var = tk.StringVar(value="No documents indexed")
        ttk.Label(main_frame, textvariable=self.index_status_var).pack(anchor=tk.W, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Search Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a canvas for scrolling
        canvas = tk.Canvas(results_frame)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        
        # Add a frame inside canvas for results
        self.results_container = ttk.Frame(canvas)
        self.results_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrolling components
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Document view
        self.doc_view_frame = ttk.LabelFrame(main_frame, text="Document Preview", padding="10")
        self.doc_view_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.doc_title_var = tk.StringVar(value="Select a search result to preview")
        ttk.Label(self.doc_view_frame, textvariable=self.doc_title_var, font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)
        
        self.doc_text = scrolledtext.ScrolledText(self.doc_view_frame, wrap=tk.WORD, height=10)
        self.doc_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_index_status(self):
        """Update the index status display."""
        if not self.client.processed_files:
            self.index_status_var.set("No documents indexed")
            return
            
        doc_count = len(self.client.processed_files)
        self.index_status_var.set(f"{doc_count} documents indexed and ready for search")
    
    def perform_search(self):
        """Execute search based on current query."""
        query = self.search_var.get().strip()
        if not query:
            self.status_var.set("Please enter a search query")
            return
        
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        # Perform search
        self.status_var.set(f"Searching for: {query}")
        
        results = self.client.query_processor.search(query)
        
        if not results:
            self.status_var.set("No matching documents found")
            ttk.Label(self.results_container, text="No results found for your query").pack(anchor=tk.W, pady=10)
            return
        
        # Display results
        for i, (doc_id, filename, score, snippet) in enumerate(results):
            self._create_result_item(i+1, doc_id, filename, score, snippet)
        
        self.status_var.set(f"Found {len(results)} results for: {query}")
    
    def _create_result_item(self, rank, doc_id, filename, score, snippet):
        """Create a UI element for a search result."""
        # Result container
        result_frame = ttk.Frame(self.results_container, style="ResultItem.TFrame")
        result_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Title with rank and filename
        title_frame = ttk.Frame(result_frame)
        title_frame.pack(fill=tk.X, anchor=tk.W)
        
        ttk.Label(title_frame, text=f"{rank}. ", font=('TkDefaultFont', 10, 'bold')).pack(side=tk.LEFT)
        filename_btn = ttk.Button(
            title_frame, 
            text=filename, 
            style="Hyperlink.TButton",
            command=lambda d=doc_id, f=filename: self._show_document(d, f)
        )
        filename_btn.pack(side=tk.LEFT)
        
        # Score display
        ttk.Label(title_frame, text=f"Score: {score:.3f}", font=('TkDefaultFont', 8)).pack(side=tk.RIGHT)
        
        # Snippet
        snippet_frame = ttk.Frame(result_frame)
        snippet_frame.pack(fill=tk.X, pady=2)
        
        snippet_text = ttk.Label(snippet_frame, text=snippet, wraplength=600)
        snippet_text.pack(anchor=tk.W)
    
    def _show_document(self, doc_id, filename):
        """Display a document in the preview pane."""
        self.doc_title_var.set(filename)
        
        # Get document text
        doc_text = self.client.query_processor.get_document(doc_id)
        
        # Display in text widget
        self.doc_text.delete(1.0, tk.END)
        self.doc_text.insert(tk.END, doc_text)
        
        # Highlight search terms if any
        query = self.search_var.get().strip().lower()
        terms = [term for term in re.split(r'\W+', query) if term and len(term) >= 2]
        
        for term in terms:
            start_pos = "1.0"
            while True:
                try:
                    start_pos = self.doc_text.search(term, start_pos, tk.END, nocase=True)
                    if not start_pos:
                        break
                    
                    end_pos = f"{start_pos}+{len(term)}c"
                    self.doc_text.tag_add("highlight", start_pos, end_pos)
                    start_pos = end_pos
                except Exception as e:
                    # Skip any search errors and continue
                    break
        
        self.doc_text.tag_configure("highlight", background="yellow")

class WorkerStatusTab:
    """GUI tab for worker status."""
    def __init__(self, parent_notebook, client):
        self.parent_notebook = parent_notebook
        self.client = client
        
        # Create tab
        self.tab = ttk.Frame(parent_notebook)
        parent_notebook.add(self.tab, text="Worker Status")
        
        self._setup_ui()
        
        # Auto-refresh flag
        self.auto_refresh = False
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main frame
        worker_frame = ttk.Frame(self.tab, padding="10")
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
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(worker_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_worker_status(self):
        """Update the worker status display."""
        try:
            self.status_var.set("Fetching worker status...")
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
            self.auto_refresh = True
            self._auto_refresh()
        else:
            # Stop auto-refresh
            self.auto_refresh = False
    
    def _auto_refresh(self):
        """Auto-refresh worker status."""
        if self.auto_refresh and self.auto_refresh_var.get():
            self.update_worker_status()
            # Schedule next refresh after 5 seconds
            self.parent_notebook.after(5000, self._auto_refresh)

class WordHiveGUI:
    """
    GUI application for the WordHive client.
    """
    def __init__(self, root, server_address="localhost", server_port=5555):
        self.root = root
        self.root.title("WordHive Client")
        self.root.geometry("1000x800")
        
        # Create client
        self.client = WordHiveClient(server_address, server_port)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create main tab
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Word Count")
        
        # Setup UI components
        self._setup_main_tab()
        
        # Selected files
        self.selected_files = []
        
        # Store the last results
        self.results = None
        
        # Add other tabs
        self.word_cloud_tab = WordCloudTab(self.notebook, self.client)
        self.query_tab = QueryTab(self.notebook, self.client)
        self.worker_tab = WorkerStatusTab(self.notebook, self.client)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set(f"Ready - Not connected to server")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_main_tab(self):
        """Set up the main tab with word count functionality."""
        # File selection
        frame_file = ttk.Frame(self.main_tab, padding="10")
        frame_file.pack(fill=tk.X)
        
        ttk.Label(frame_file, text="Select text files:").pack(side=tk.LEFT)
        self.selected_files_var = tk.StringVar()
        self.selected_files_var.set("No files selected")
        ttk.Label(frame_file, textvariable=self.selected_files_var).pack(side=tk.LEFT, padx=10)
        ttk.Button(frame_file, text="Browse...", command=self.select_files).pack(side=tk.RIGHT)
        
        # Server connection
        frame_connection = ttk.Frame(self.main_tab, padding="10")
        frame_connection.pack(fill=tk.X)
        
        ttk.Label(frame_connection, text="Server:").pack(side=tk.LEFT)
        self.server_var = tk.StringVar(value=self.client.server_address)
        ttk.Entry(frame_connection, textvariable=self.server_var, width=20).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame_connection, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_var = tk.StringVar(value=str(self.client.server_port))
        ttk.Entry(frame_connection, textvariable=self.port_var, width=6).pack(side=tk.LEFT)
        
        ttk.Button(frame_connection, text="Connect", command=self.connect_to_server).pack(side=tk.LEFT, padx=10)
        
        # Processing options
        frame_options = ttk.Frame(self.main_tab, padding="10")
        frame_options.pack(fill=tk.X)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress = ttk.Progressbar(frame_options, orient="horizontal", variable=self.progress_var)
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(frame_options, text="Process Files", command=self.process_files).pack(side=tk.RIGHT)
        
        # Results area
        frame_results = ttk.LabelFrame(self.main_tab, text="Processing Results", padding="10")
        frame_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(frame_results, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization frame
        self.frame_viz = ttk.LabelFrame(self.main_tab, text="Word Frequency Visualization", padding="10")
        self.frame_viz.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def select_files(self):
        """Open file dialog to select text files."""
        filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        self.selected_files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if self.selected_files:
            self.selected_files_var.set(f"{len(self.selected_files)} files selected")
        else:
            self.selected_files_var.set("No files selected")
    
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
            self.client.disconnect()
            
            # Update client settings
            self.client.server_address = server
            self.client.server_port = port
            
            # Connect to server
            if self.client.connect():
                self.status_var.set(f"Connected to server at {server}:{port}")
                # Update worker status tab
                self.worker_tab.update_worker_status()
            else:
                self.status_var.set("Failed to connect to server")
                messagebox.showerror("Connection Error", "Failed to connect to server")
        except Exception as e:
            self.status_var.set(f"Connection error: {str(e)}")
            messagebox.showerror("Connection Error", str(e))
    
    def process_files(self):
        """Process the selected files using the distributed system."""
        if not self.selected_files:
            messagebox.showerror("Error", "No files selected")
            return
        
        if not self.client.connected:
            messagebox.showerror("Error", "Not connected to server")
            return
        
        # Update status
        self.status_var.set("Sending files to server...")
        self.progress_var.set(0)
        self.root.update()
        
        # Process files in a separate thread to avoid UI freezing
        threading.Thread(target=self._do_processing, daemon=True).start()
    
    def _do_processing(self):
        """Perform file processing in a background thread."""
        try:
            # Send files to server
            results = self.client.process_files(self.selected_files)
            
            if results:
                # Store results
                self.results = results
                
                # Update UI from main thread
                self.root.after(0, lambda: self._display_results(results))
                self.root.after(0, lambda: self.status_var.set("Processing complete"))
                self.root.after(0, lambda: self.progress_var.set(100))
                
                # Update search tab
                self.root.after(0, lambda: self.query_tab.update_index_status())
            else:
                self.root.after(0, lambda: self.status_var.set("Processing failed"))
                self.root.after(0, lambda: messagebox.showerror("Processing Error", "Failed to process files"))
        except Exception as e:
            error_msg = f"Error processing files: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Processing Error", error_msg))
    
    def _display_results(self, results):
        """Display processing results."""
        if not results:
            return
        
        # Extract counts
        word_counts = results.get('all_counts', {})
        total_words = results.get('total_words', 0)
        unique_words = results.get('unique_words', 0)
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Processing completed\n\n")
        self.results_text.insert(tk.END, f"Total words: {total_words}\n")
        self.results_text.insert(tk.END, f"Total unique words: {unique_words}\n\n")
        self.results_text.insert(tk.END, "Top 20 most common words:\n")
        
        # Display top words
        for word, count in Counter(word_counts).most_common(20):
            self.results_text.insert(tk.END, f"{word}: {count}\n")
        
        # Update word cloud tab
        if hasattr(self.word_cloud_tab, 'update_word_counts'):
            self.word_cloud_tab.update_word_counts(word_counts)
        
        # Visualize results
        self._create_visualization(word_counts)
    
    def _create_visualization(self, word_counts):
        """Create visualization of word count results."""
        # Clear previous visualization
        for widget in self.frame_viz.winfo_children():
            widget.destroy()
        
        # Get top 10 words
        top_words = dict(Counter(word_counts).most_common(10))
        words = list(top_words.keys())
        counts = list(top_words.values())
        
        # Create figure with bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
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

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='WordHive Client')
    parser.add_argument('--server', default='localhost', help='Server address')
    parser.add_argument('--port', type=int, default=5555, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger('WordHiveExternalClient').setLevel(logging.DEBUG)
    
    # Create and run GUI
    root = tk.Tk()
    app = WordHiveGUI(root, args.server, args.port)
    root.mainloop()

if __name__ == "__main__":
    main()