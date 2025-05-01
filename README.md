# WordHive: Distributed Word Count System

WordHive is a scalable distributed system for processing and analyzing text files using parallel computing techniques. It employs a server-worker-client architecture to distribute text processing workloads across multiple nodes, providing efficient word frequency analysis and visualization.

## 🌟 Features

- **Distributed Processing**: Parallel text processing across multiple worker nodes
- **Interactive UI**: Sleek GUI with real-time processing feedback
- **Rich Visualizations**: Word frequency charts and customizable word clouds
- **Advanced Search**: Full-text search with context-aware results and highlighting
- **Worker Management**: Real-time monitoring of worker nodes with status tracking
- **Fault Tolerance**: Graceful handling of worker disconnections
- **Horizontal Scaling**: Add more workers at runtime to increase processing capacity

## 🏗️ Architecture

The system consists of three main components:

### Server
- Central coordinator for the entire system
- Manages client connections and worker registration
- Splits input text into chunks for distributed processing
- Schedules tasks and tracks progress
- Aggregates results from workers

### Worker Nodes
- Connect to the server and register for work
- Process assigned text chunks
- Calculate word frequencies
- Send results back to the server
- Provide heartbeat signals for health monitoring

### Client Applications
- Feature-rich GUI for text processing and visualization
- File selection interface for multiple document processing
- Real-time worker status monitoring
- Search functionality across processed documents
- Word cloud generation with customization options

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Required packages: `zmq`, `matplotlib`, `tkinter`, `wordcloud` (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/wordhive.git
   cd wordhive
   ```

2. Install dependencies:
   ```bash
   python setup.py
   ```

### Running the System

1. Start the server (central coordinator):
   ```bash
   python server.py
   ```

2. Start worker nodes (can be on different machines):
   ```bash
   python worker_manager.py start <number_of_workers> --server <server_ip_address>
   ```

3. Launch the client application:
   ```bash
   python client.py --server <server_ip_address>
   ```

4. For external client (simulating access outside internal network):
   ```bash
   python external_client.py --server <server_ip_address>
   ```

## 💻 Usage

### Basic Workflow

1. Connect to server using the GUI interface
2. Select text files for processing
3. Process files with one click
4. View word frequency results and visualizations
5. Search across all processed documents
6. Monitor worker status and system performance

### Command Line Options

**Client:**
```bash
python client.py --server <address> --port <port> [--debug]
```

**Worker:**
```bash
python worker.py --server <address> --worker-port <port> --heartbeat-port <port> [--debug]
```

**Worker Manager:**
```bash
python worker_manager.py (start|stop|status|monitor) [count] --server <address> [--interval <seconds>]
```

## 📊 Performance

The system's performance scales linearly with the addition of worker nodes. Adding more workers increases processing speed for large documents.

## 🔧 Implementation Details

- **Communication Protocol**: TCP sockets with ZeroMQ for reliable messaging
- **Message Format**: JSON for data exchange between components
- **Concurrency**: Multi-threaded server design with separate threads for client handling, worker communication, task scheduling, and heartbeat monitoring
- **Fault Handling**: Dead worker detection and task rescheduling
- **Load Balancing**: Even distribution of tasks among available workers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Halim Tiberino**
- **Brian Chukwuisiocha**
- **Victoria Hinton**

## 🎓 Acknowledgements

- Developed as a project for the Parallel & Distributed Programming course
- Inspired by distributed computing frameworks like Hadoop
