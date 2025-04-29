# WordHive Client

A simple client for connecting to our class WordHive server for distributed text processing.

## What is WordHive?

WordHive is a distributed text processing system that can analyze large text files by splitting the work across multiple computers. The system counts word frequencies, generates visualizations, and provides search capabilities.

## Requirements

- Python 3.6 or higher
- Required packages: `zmq`, `matplotlib`, `tqdm`

## Installation

1. Download the `wordhive_client.py` file
2. Install the required packages:

```bash
pip install pyzmq matplotlib tqdm
```

## Usage

To use the client, run it from the command line with the server address and the text files you want to process:

```bash
python wordhive_client.py --server SERVER_ADDRESS [--port PORT] file1.txt file2.txt file3.txt
```

Replace:
- `SERVER_ADDRESS` with the address provided by your instructor (e.g., `classroom.example.edu`)
- `PORT` with the port number (default is 5555)
- `file1.txt file2.txt file3.txt` with the text files you want to process

## Example

```bash
python wordhive_client.py --server classroom.example.edu shakespeare.txt moby-dick.txt
```

This will:
1. Connect to the WordHive server running at `classroom.example.edu`
2. Upload your text files for processing
3. Wait for the server to process your files
4. Display the results including:
   - Total word count
   - Unique word count
   - Top 20 most common words
   - A visualization of the top 10 words

## Troubleshooting

1. If you get a "Connection refused" error:
   - Make sure the server is running
   - Check that you're using the correct server address and port
   - Ensure your network allows connections to the specified port

2. If you get a timeout error:
   - The server might be busy or not responding
   - Try again later or with smaller files

3. If you can't install the required packages:
   - Try using a virtual environment:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     pip install pyzmq matplotlib tqdm
     ```
