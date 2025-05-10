import logging

# ANSI color codes
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    ENDC = '\033[0m'

# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    def __init__(self, color):
        super().__init__('%(name)s: %(message)s')
        self.color = color

    def format(self, record):
        record.msg = f"{self.color}{record.msg}{Colors.ENDC}"
        return super().format(record)

def setup_logger(name, color):
    """Set up a single logger with the specified color."""
    logger = logging.getLogger(name)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Add new handler
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(color))
    logger.addHandler(handler)
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def setup_loggers():
    """Set up all loggers with colors and appropriate levels."""
    # Configure root logger to not propagate
    logging.getLogger().handlers = []
    logging.getLogger().propagate = False
    
    # Set up individual loggers
    main_logger = setup_logger('main', Colors.CYAN)
    exr_logger = setup_logger('exr', Colors.GREEN)
    raw_logger = setup_logger('raw', Colors.YELLOW)
    fusion_logger = setup_logger('fusion', Colors.BLUE)
    exif_logger = setup_logger('exif', Colors.MAGENTA)
    weight_logger = setup_logger('weight', Colors.RED)
    metadata_logger = setup_logger('metadata', Colors.MAGENTA)  # Using magenta for metadata like exif

    return {
        'main': main_logger,
        'exr': exr_logger,
        'raw': raw_logger,
        'fusion': fusion_logger,
        'exif': exif_logger,
        'weight': weight_logger,
        'metadata': metadata_logger
    } 