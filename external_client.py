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
import re
import math
from collections import Counter, defaultdict
import heapq
from tqdm import tqdm

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
                    print(f"Error reading {filepath}: {str(e)}")
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
        
        # Clear previous processed files
        self.processed_files = []
        self.file_content_map = {}
        
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
                
                # Store file content for search indexing
                self.file_content_map[file_path] = content
                self.processed_files.append(file_path)
                
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
                
                # Build search index in background
                if results:
                    self._build_search_index()
                
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
    
    def _build_search_index(self):
        """Build search index from processed files."""
        if not self.processed_files:
            print("No files to index")
            return
        
        print("\nBuilding search index...")
        try:
            # Build index in background thread to avoid blocking
            threading.Thread(
                target=self._do_indexing, 
                daemon=True
            ).start()
        except Exception as e:
            print(f"Error starting indexing thread: {str(e)}")
    
    def _do_indexing(self):
        """Perform indexing in a background thread."""
        try:
            indexed_count = self.query_processor.index_documents(
                self.processed_files, 
                self.file_content_map
            )
            print(f"Indexed {indexed_count} documents for search")
        except Exception as e:
            print(f"Error building index: {str(e)}")
    
    def search(self, query, max_results=10):
        """
        Search for documents matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not query:
            print("No query specified")
            return []
        
        if not self.processed_files:
            print("No documents indexed for search")
            return []
        
        # Execute search
        print(f"Searching for: {query}")
        results = self.query_processor.search(query, max_results)
        
        if not results:
            print("No results found")
            return []
        
        # Format and return results
        formatted_results = []
        
        for i, (doc_id, filename, score, snippet) in enumerate(results):
            formatted_results.append({
                'rank': i + 1,
                'filename': filename,
                'score': score,
                'snippet': snippet,
                'doc_id': doc_id
            })
        
        return formatted_results
    
    def get_document(self, doc_id):
        """
        Get the full text of a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document text and filename
        """
        # Get document path
        doc_path = self.query_processor.inverted_index.get_document_path(doc_id)
        if not doc_path:
            return None, "Document not found"
        
        # Get document filename
        filename = os.path.basename(doc_path)
        
        # Get document text
        doc_text = self.query_processor.get_document(doc_id)
        
        return filename, doc_text
    
    def display_search_results(self, results):
        """
        Display search results in a formatted way.
        
        Args:
            results: List of search result dictionaries
        """
        if not results:
            print("No search results to display")
            return
        
        print("\n===== Search Results =====")
        for result in results:
            print(f"{result['rank']}. {result['filename']} (Score: {result['score']:.3f})")
            print(f"   {result['snippet']}")
            print()
    
    def display_document(self, doc_id):
        """
        Display the full text of a document.
        
        Args:
            doc_id: Document identifier
        """
        filename, doc_text = self.get_document(doc_id)
        
        if not filename or not doc_text:
            print("Document not found")
            return
        
        print(f"\n===== {filename} =====")
        print(doc_text)
    
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
    
    # Add search functionality arguments
    parser.add_argument('--search', help='Search query')
    parser.add_argument('--view-doc', type=int, help='View document by ID')
    parser.add_argument('--max-results', type=int, default=10, help='Maximum search results')
    
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
                
                # If search query is provided, execute search
                if args.search:
                    print(f"\nExecuting search for: {args.search}")
                    search_results = client.search(args.search, args.max_results)
                    client.display_search_results(search_results)
                    
                    # Ask if user wants to view a full document
                    if search_results:
                        print("\nTo view a full document, use the --view-doc parameter with the document ID")
                
                # If view-doc is provided, display the document
                if args.view_doc is not None:
                    client.display_document(args.view_doc)
        
        # Handle search without processing files first
        elif args.search:
            print("Please process files before searching. Provide file paths to process.")
            print("Example: python wordhive_client.py --server classroom.example.edu file1.txt file2.txt --search 'query'")
        
        # Handle view-doc without processing files first
        elif args.view_doc is not None:
            print("Please process files before viewing documents. Provide file paths to process.")
            print("Example: python wordhive_client.py --server classroom.example.edu file1.txt file2.txt --view-doc 1")
        
        else:
            print("No files specified. Use the --help option for usage information.")
            print("Example: python wordhive_client.py --server classroom.example.edu file1.txt file2.txt")
            print("         python wordhive_client.py --server classroom.example.edu file1.txt file2.txt --search 'query'")
            print("         python wordhive_client.py --server classroom.example.edu file1.txt file2.txt --view-doc 1")
            
    finally:
        # Disconnect from server
        client.disconnect()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())