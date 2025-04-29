#!/usr/bin/env python3
# setup.py
"""
Setup script for the distributed word count system.
"""
import os
import sys
import subprocess
import platform

# Required packages
REQUIRED_PACKAGES = [
    "pyzmq",          # ZeroMQ messaging library
    "matplotlib",     # For visualizations
    "wordcloud",      # For word cloud generation
    "tabulate",       # For pretty printing tables
]

def check_python_version():
    """Check if Python version is compatible."""
    min_version = (3, 6)
    current_version = sys.version_info
    
    if current_version < min_version:
        print(f"Error: Python {min_version[0]}.{min_version[1]} or higher is required")
        print(f"Current version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    print(f"Python version {current_version[0]}.{current_version[1]}.{current_version[2]} is compatible")
    return True

def install_dependencies():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + REQUIRED_PACKAGES)
        print("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_startup_scripts():
    """Create startup scripts for different platforms."""
    print("Creating startup scripts...")
    
    # Determine platform
    system = platform.system()
    
    if system == "Windows":
        # Create Windows batch files
        with open("start_server.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting distributed word count server...\n")
            f.write(f"python server.py %*\n")
            f.write("pause\n")
        
        with open("start_worker.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting worker node...\n")
            f.write(f"python worker.py %*\n")
            f.write("pause\n")
        
        with open("start_client.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting distributed word count client...\n")
            f.write(f"python client.py %*\n")
        
        with open("worker_manager.bat", "w") as f:
            f.write("@echo off\n")
            f.write(f"python worker_manager.py %*\n")
            f.write("pause\n")
        
        print("Created Windows batch files")
        
    else:
        # Create Unix shell scripts
        with open("start_server.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo Starting distributed word count server...\n")
            f.write(f"python3 server.py \"$@\"\n")
        
        with open("start_worker.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo Starting worker node...\n")
            f.write(f"python3 worker.py \"$@\"\n")
        
        with open("start_client.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo Starting distributed word count client...\n")
            f.write(f"python3 client.py \"$@\"\n")
        
        with open("worker_manager.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"python3 worker_manager.py \"$@\"\n")
        
        # Make scripts executable
        for script in ["start_server.sh", "start_worker.sh", "start_client.sh", "worker_manager.sh"]:
            os.chmod(script, 0o755)
        
        print("Created Unix shell scripts and made them executable")
    
    return True

def main():
    """Main setup function."""
    print("=== Distributed Word Count System Setup ===")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Create startup scripts
    if not create_startup_scripts():
        return 1
    
    print("\nSetup completed successfully!")
    print("\nTo get started:")
    print("1. Start the server: start_server.sh/bat")
    print("2. Start workers: worker_manager.sh/bat start <count>")
    print("3. Start the client: start_client.sh/bat")
    print("\nFor more information, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())