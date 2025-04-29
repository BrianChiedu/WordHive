# wordcloud_enhancement.py
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud, STOPWORDS
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog
import io
import os
import threading
from collections import Counter

class WordCloudGenerator:
    """
    Handles the generation of word clouds from word frequency data.
    """
    def __init__(self):
        # Default settings
        self.width = 800
        self.height = 600
        self.background_color = "white"
        self.colormap = "viridis"
        self.max_words = 200
        self.min_font_size = 4
        self.max_font_size = None  # Auto-determined
        self.random_state = 42  # For reproducibility
        self.mask = None
        self.stopwords = set(STOPWORDS)
        self.contour_width = 1
        self.contour_color = 'steelblue'
        
    def set_mask(self, mask_path):
        """Set a custom mask/shape for the word cloud."""
        try:
            mask_img = np.array(Image.open(mask_path))
            self.mask = mask_img
            return True
        except Exception as e:
            print(f"Error loading mask image: {e}")
            self.mask = None
            return False
    
    def generate_wordcloud(self, word_counts):
        """Generate a word cloud from word frequency data."""
        # Create WordCloud object with current settings
        wc = WordCloud(
            background_color=self.background_color,
            max_words=self.max_words,
            mask=self.mask,
            colormap=self.colormap,
            width=self.width,
            height=self.height,
            min_font_size=self.min_font_size,
            max_font_size=self.max_font_size,
            stopwords=self.stopwords,
            contour_width=self.contour_width,
            contour_color=self.contour_color,
            random_state=self.random_state
        )
        
        # Generate from word frequency dictionary
        wc.generate_from_frequencies(word_counts)
        
        return wc
    
    def create_figure(self, wordcloud):
        """Create a matplotlib figure with the wordcloud."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout()
        return fig
    
    def save_wordcloud(self, wordcloud, output_path, format="png", dpi=300):
        """Save the wordcloud to an image file."""
        try:
            wordcloud.to_file(output_path)
            return True
        except Exception as e:
            print(f"Error saving wordcloud: {e}")
            return False

class WordCloudTab:
    """
    GUI tab for Word Cloud visualization and customization.
    """
    def __init__(self, parent_notebook, word_processor=None):
        """
        Initialize the Word Cloud tab.
        
        Args:
            parent_notebook: The parent ttk.Notebook widget
            word_processor: Reference to the main file processor
        """
        self.parent_notebook = parent_notebook
        self.word_processor = word_processor
        self.generator = WordCloudGenerator()
        self.current_wordcloud = None
        self.last_word_counts = None
        
        # Create tab
        self.tab = ttk.Frame(parent_notebook)
        parent_notebook.add(self.tab, text="Word Cloud")
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main frame
        main_frame = ttk.Frame(self.tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Word Cloud Settings", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Right panel for visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Word cloud display area
        self.wordcloud_frame = ttk.Frame(viz_frame)
        self.wordcloud_frame.pack(fill=tk.BOTH, expand=True)
        
        # Message label
        self.message_var = tk.StringVar(value="Process text files first to generate a word cloud")
        ttk.Label(self.wordcloud_frame, textvariable=self.message_var, font=('TkDefaultFont', 12)).pack(pady=50)
        
        # Control elements
        # Background color - Use a standard tk.Frame instead of ttk.Frame for color display
        ttk.Label(control_frame, text="Background Color:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.bg_color_frame = tk.Frame(control_frame, width=30, height=20, relief=tk.SUNKEN)
        self.bg_color_frame.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        self.bg_color_frame.config(background=self.generator.background_color)
        ttk.Button(control_frame, text="Choose...", command=self.choose_bg_color, width=10).grid(row=0, column=2, padx=5, pady=2)
        
        # Color map
        ttk.Label(control_frame, text="Color Scheme:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                     'Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 
                     'RdYlBu', 'RdBu', 'coolwarm', 'rainbow', 'jet']
        self.colormap_var = tk.StringVar(value=self.generator.colormap)
        ttk.Combobox(control_frame, textvariable=self.colormap_var, values=color_maps, width=15).grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Max words
        ttk.Label(control_frame, text="Maximum Words:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_words_var = tk.IntVar(value=self.generator.max_words)
        ttk.Spinbox(control_frame, from_=50, to=500, increment=50, textvariable=self.max_words_var, width=10).grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Min font size
        ttk.Label(control_frame, text="Min Font Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_font_var = tk.IntVar(value=self.generator.min_font_size)
        ttk.Spinbox(control_frame, from_=1, to=20, textvariable=self.min_font_var, width=10).grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Custom mask/shape
        ttk.Label(control_frame, text="Custom Shape:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.mask_path_var = tk.StringVar(value="None")
        ttk.Label(control_frame, textvariable=self.mask_path_var, width=20).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Button(control_frame, text="Browse...", command=self.select_mask, width=10).grid(row=4, column=2, padx=5, pady=2)
        
        # Stopwords
        ttk.Label(control_frame, text="Remove Common Words:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.use_stopwords_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, variable=self.use_stopwords_var).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Exclude words below frequency
        ttk.Label(control_frame, text="Min Word Frequency:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_freq_var = tk.IntVar(value=1)
        ttk.Spinbox(control_frame, from_=1, to=100, textvariable=self.min_freq_var, width=10).grid(row=6, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Generate Word Cloud", command=self.generate_cloud).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Image", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def choose_bg_color(self):
        """Open color chooser dialog for background color."""
        color = colorchooser.askcolor(initialcolor=self.generator.background_color, 
                                   title="Choose Background Color")[1]
        if color:
            self.generator.background_color = color
            self.bg_color_frame.config(background=color)
    
    def select_mask(self):
        """Open file dialog to select a mask image."""
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp")]
        mask_path = filedialog.askopenfilename(filetypes=filetypes, 
                                            title="Select Shape Image")
        
        if mask_path:
            filename = os.path.basename(mask_path)
            self.mask_path_var.set(filename)
            
            if self.generator.set_mask(mask_path):
                self.status_var.set(f"Shape loaded: {filename}")
            else:
                self.status_var.set(f"Error loading shape: {filename}")
                self.mask_path_var.set("None")
    
    def update_word_counter(self, word_processor):
        """Update the reference to the word processor."""
        self.word_processor = word_processor
    
    def update_word_counts(self, word_counts):
        """Update the current word counts."""
        self.last_word_counts = word_counts
        # Show notification that word cloud is ready to be generated
        self.message_var.set("Word counts updated. Click 'Generate Word Cloud' to create visualization.")
    
    def generate_cloud(self):
        """Generate and display the word cloud."""
        if not self.last_word_counts:
            if hasattr(self.word_processor, 'results'):
                # Try to get results from the word counter
                if not self.word_processor.results:
                    self.message_var.set("No word count data available. Process files first.")
                    return
                
                # Aggregate results
                self.last_word_counts = Counter()
                for _, counts in self.word_processor.results:
                    self.last_word_counts.update(counts)
            else:
                self.message_var.set("No word count data available. Process files first.")
                return
        
        # Apply minimum frequency filter
        min_freq = self.min_freq_var.get()
        filtered_words = {word: count for word, count in self.last_word_counts.items() 
                          if count >= min_freq}
        
        if not filtered_words:
            self.status_var.set("No words meet the minimum frequency criteria")
            return
        
        # Update generator settings
        self.generator.colormap = self.colormap_var.get()
        self.generator.max_words = self.max_words_var.get()
        self.generator.min_font_size = self.min_font_var.get()
        
        # Handle stopwords
        if self.use_stopwords_var.get():
            self.generator.stopwords = set(STOPWORDS)
        else:
            self.generator.stopwords = set()
        
        # Generate in a separate thread to avoid UI freezing
        self.status_var.set("Generating word cloud...")
        threading.Thread(target=self._do_generate, args=(filtered_words,), daemon=True).start()
    
    def _do_generate(self, word_counts):
        """Perform word cloud generation in a background thread."""
        try:
            # Generate word cloud
            self.current_wordcloud = self.generator.generate_wordcloud(word_counts)
            
            # Update UI from main thread
            self.parent_notebook.after(0, self._update_display)
            
        except Exception as e:
            error_msg = f"Error generating word cloud: {str(e)}"
            self.parent_notebook.after(0, lambda: self.status_var.set(error_msg))
    
    def _update_display(self):
        """Update the UI with the new word cloud."""
        if not self.current_wordcloud:
            return
            
        # Clear previous display
        for widget in self.wordcloud_frame.winfo_children():
            widget.destroy()
        
        # Create figure with word cloud
        fig = self.generator.create_figure(self.current_wordcloud)
        
        # Display in canvas
        canvas = FigureCanvasTkAgg(fig, master=self.wordcloud_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update status
        word_count = len(self.current_wordcloud.words_)
        self.status_var.set(f"Word cloud generated with {word_count} words")
    
    def save_image(self):
        """Save the current word cloud as an image file."""
        if not self.current_wordcloud:
            self.status_var.set("No word cloud to save. Generate one first.")
            return
        
        # Open file dialog
        filetypes = [
            ("PNG Image", "*.png"),
            ("JPEG Image", "*.jpg"),
            ("BMP Image", "*.bmp"),
            ("TIFF Image", "*.tiff")
        ]
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            title="Save Word Cloud"
        )
        
        if filename:
            if self.generator.save_wordcloud(self.current_wordcloud, filename):
                self.status_var.set(f"Word cloud saved to {filename}")
            else:
                self.status_var.set("Error saving word cloud")