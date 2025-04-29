# query_processor.py
import re
import os
import math
from collections import defaultdict, Counter
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import heapq

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
    
    def index_processed_results(self, processor, filepaths):
        """
        Build inverted index from processed documents.
        
        Args:
            processor: The file processor object with results
            filepaths: List of file paths corresponding to results
        """
        self.is_indexing = True
        
        # Map file paths to document IDs
        doc_paths = {}
        for i, path in enumerate(filepaths):
            doc_paths[i] = path
        
        # Add documents to index
        for doc_id, filepath in doc_paths.items():
            chunks = {}
            self.inverted_index.add_document(doc_id, filepath, chunks)
        
        # Process results from processor
        if hasattr(processor, 'file_chunks') and hasattr(processor, 'results'):
            # Build a map from chunk_id to file chunks
            chunk_map = {chunk_id: text for text, chunk_id in processor.file_chunks}
            
            # Group results by document ID
            doc_results = defaultdict(list)
            
            for chunk_id, word_counts in processor.results:
                # Determine which document this chunk belongs to
                # Check if processor has a file_map attribute
                if hasattr(processor, 'file_map') and chunk_id in processor.file_map:
                    # Get the filepath for this chunk
                    filepath = processor.file_map[chunk_id]
                    # Find the document ID for this filepath
                    doc_id = None
                    for d_id, path in doc_paths.items():
                        if path == filepath:
                            doc_id = d_id
                            break
                else:
                    # Simple mapping, may need adjustment for your specific structure
                    doc_id = chunk_id % len(filepaths)
                
                if doc_id is not None:
                    doc_results[doc_id].append((chunk_id, word_counts))
                    
                    # Add this chunk's results to the index
                    chunk_text = chunk_map.get(chunk_id, "")
                    self.inverted_index.add_chunk_results(doc_id, chunk_id, word_counts, chunk_text)
        
        # Prepare index for searching
        self.inverted_index.prepare_index()
        
        self.is_indexing = False
    
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


class QueryTab:
    """
    GUI tab for search functionality.
    """
    def __init__(self, parent_notebook, word_processor=None):
        """
        Initialize the Query tab.
        
        Args:
            parent_notebook: The parent ttk.Notebook widget
            word_processor: Reference to the main file processor
        """
        self.parent_notebook = parent_notebook
        self.word_processor = word_processor
        self.query_processor = QueryProcessor()
        self.filepaths = []
        
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
    
    def update_processor(self, word_processor, filepaths):
        """
        Update the reference to the word processor and rebuild the index.
        
        Args:
            word_processor: The updated file processor
            filepaths: List of file paths that were processed
        """
        self.word_processor = word_processor
        self.filepaths = filepaths
        
        # Clear any existing results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        # Build index in a separate thread
        self.status_var.set("Building search index...")
        self.index_status_var.set("Indexing documents...")
        
        threading.Thread(target=self._do_indexing, daemon=True).start()
    
    def _do_indexing(self):
        """Perform indexing in a background thread."""
        try:
            self.query_processor.index_processed_results(self.word_processor, self.filepaths)
            
            # Update UI from main thread
            doc_count = self.query_processor.inverted_index.total_docs
            self.parent_notebook.after(0, lambda: self.index_status_var.set(
                f"{doc_count} documents indexed and ready for search"))
            self.parent_notebook.after(0, lambda: self.status_var.set("Ready"))
            
        except Exception as e:
            error_msg = f"Error building index: {str(e)}"
            self.parent_notebook.after(0, lambda: self.status_var.set(error_msg))
    
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
        
        results = self.query_processor.search(query)
        
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
        doc_text = self.query_processor.get_document(doc_id)
        
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