import logging
import sys
import os 

# --- Configuration Constants ---
LOG_FILE = os.path.join(os.getcwd(), 'app.log') 
LOG_LEVEL = logging.INFO

def get_logger(name: str = "SCIMILARITY_PROJECT") -> logging.Logger:
    """
    Creates and returns a configured logger instance that prints to console 
    and writes to a log file, including detailed format.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    # 1. Define the Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(filename)s:%(funcName)s:%(lineno)d | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 2. Console Handler (Ensures output is printed to the terminal)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 3. File Handler (Ensures logs are written to a file)
    # Ensure the directory for the log file exists if it's placed outside the current dir
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logger initialized successfully.")
    
    return logger
