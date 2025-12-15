"""
Logging utilities for K-1 Self-Learning System.
"""

import os
import sys
from datetime import datetime
from pathlib import Path


class Logger:
    """Simple logger for training progress."""

    def __init__(self, log_dir: str = 'logs', name: str = 'k1_training'):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            name: Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{name}_{timestamp}.log'

        self.file_handle = open(self.log_file, 'w')

    def log(self, message: str, level: str = 'INFO', print_console: bool = True):
        """
        Log a message.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
            print_console: Whether to also print to console
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] [{level}] {message}"

        self.file_handle.write(formatted + '\n')
        self.file_handle.flush()

        if print_console:
            print(formatted)

    def info(self, message: str):
        """Log info message."""
        self.log(message, 'INFO')

    def warning(self, message: str):
        """Log warning message."""
        self.log(message, 'WARNING')

    def error(self, message: str):
        """Log error message."""
        self.log(message, 'ERROR')

    def section(self, title: str):
        """Log a section header."""
        separator = '=' * 60
        self.log(separator, print_console=True)
        self.log(title, print_console=True)
        self.log(separator, print_console=True)

    def close(self):
        """Close the log file."""
        self.file_handle.close()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'file_handle') and self.file_handle:
            self.file_handle.close()
