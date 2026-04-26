import inspect
import logging
logging.captureWarnings(True)
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

import colorlog

_ACTIVE_LOGGER_NAME = 'SynapseLogger'


def set_active_logger_name(name: str) -> None:
    global _ACTIVE_LOGGER_NAME
    _ACTIVE_LOGGER_NAME = name


def get_active_logger_name() -> str:
    return _ACTIVE_LOGGER_NAME


def get_logger(name_suffix: str | None = None) -> logging.Logger:
    base_name = get_active_logger_name()
    if name_suffix:
        return logging.getLogger(f"{base_name}.{name_suffix}")
    return logging.getLogger(base_name)


class LoggerProxy:
    """
    Lazily resolve the currently active Synapse logger.

    This allows module-level ``_logger`` objects to follow the runtime logger
    name configured through ``EnhancedLogger(name=...)`` .
    """

    def __init__(self, name_suffix: str | None = None):
        self.name_suffix = name_suffix

    @property
    def logger(self) -> logging.Logger:
        return get_logger(self.name_suffix)

    def __getattr__(self, item):
        return getattr(self.logger, item)


def _parse_level(level: Union[int, str]) -> int:
    """
    Parses the logging level from a string or returns the integer level.
    """
    if isinstance(level, str):
        return getattr(logging, level.upper())
    return level


class EnhancedLogger:
    """
    A logger that supports console output and file logging with rotation.
    It can log to a main log file and a debug log file, with options for
    enabling/disabling console output and setting log file sizes.
    """
    def __init__(
            self,
            name: str = 'SynapseLogger',
            enable_stdout: bool = True,
            ignore_python_warnings: bool = False,
            log_file: Optional[str] = None,
            debug_file: Optional[str] = None,
            logger_level: Union[int, str] = logging.DEBUG,
            console_level: Union[int, str] = logging.INFO,
            file_level: Union[int, str] = logging.INFO,
            debug_level: Union[int, str] = logging.DEBUG,
            max_bytes: int = 32 * 1024 * 1024,
            backup_count: int = 5
    ):
        """
        Initializes the EnhancedLogger with specified parameters.

        Args:
            name (str): The name of the logger.
            enable_stdout (bool): Whether to enable console output. Default is True.
            ignore_python_warnings (bool): Whether to ignore Python warnings. Default is False.
            log_file (Optional[str]): Path to the main log file. If None, no file logging is done.
            debug_file (Optional[str]): Path to the debug log file. If None, no debug logging is done.
            logger_level (Union[int, str]): Logging level for the logger. Default is DEBUG.
            console_level (Union[int, str]): Logging level for console output. Default is INFO.
            file_level (Union[int, str]): Logging level for the main log file. Default is INFO.
            debug_level (Union[int, str]): Logging level for the debug log file. Default is DEBUG.
            max_bytes (int): Maximum size of each log file before rotation. Default is 32MB.
            backup_count (int): Number of backup files to keep. Default is 5.
        """
        set_active_logger_name(name)
        self.logger = get_logger()
        self.ignore_python_warnings = ignore_python_warnings
        self.handlers = []
        self._bridged_logger_names: list[str] = []
        self._configure_logger(
            enable_stdout,
            log_file,
            debug_file,
            _parse_level(logger_level),
            _parse_level(console_level),
            _parse_level(file_level),
            _parse_level(debug_level),
            max_bytes,
            backup_count
        )

    def _configure_logger(
            self,
            enable_stdout: bool,
            log_file: Optional[str],
            debug_file: Optional[str],
            logger_level: int,
            console_level: int,
            file_level: int,
            debug_level: int,
            max_bytes: int,
            backup_count: int
    ):
        # clear existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # set the logger level
        self.logger.setLevel(logger_level)

        # console output configuration (colorized)
        if enable_stdout:
            console_format = colorlog.ColoredFormatter(
                '%(log_color)s[%(asctime)s] %(levelname)-8s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                reset=True,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            console_handler.setFormatter(console_format)
            self.handlers.append(console_handler)

        # main log file configuration
        if log_file:
            file_format = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(file_format)
            self.handlers.append(file_handler)

        # debug log file configuration
        if debug_file:
            debug_format = logging.Formatter(
                '[%(asctime)s] [%(module)s:%(funcName)s:%(lineno)d] %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            debug_handler = RotatingFileHandler(
                filename=debug_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            debug_handler.setLevel(debug_level)
            debug_handler.setFormatter(debug_format)
            self.handlers.append(debug_handler)

        # Add all handlers to the logger
        for handler in self.handlers:
            self.logger.addHandler(handler)

    @classmethod
    def from_config(cls, cfg: dict):
        """
        Creates an instance of EnhancedLogger from a configuration dictionary.
        """
        sig = inspect.signature(cls.__init__)
        params = {}

        for param in list(sig.parameters.values())[1:]:
            if param.name in cfg:
                params[param.name] = cfg[param.name]
            elif param.default != param.empty:
                params[param.name] = param.default
            else:
                raise ValueError(f"Missing required parameter '{param.name}' for logger configuration.")

        return cls(**params)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.
        """
        return self.logger

    def bridge_loggers(self, logger_names: list[str], propagate: bool = False) -> None:
        """
        Attach this logger's handlers to other named loggers.

        This is useful for third-party packages that log with their own logger names
        but should write into the same Synapse log files.
        """
        logger_names = list(dict.fromkeys(logger_names))

        if self.ignore_python_warnings and 'py.warnings' in logger_names:
            logger_names.remove('py.warnings')

        for logger_name in logger_names:
            external_logger = logging.getLogger(logger_name)
            external_logger.handlers.clear()
            external_logger.setLevel(self.logger.level)
            external_logger.propagate = propagate
            for handler in self.handlers:
                external_logger.addHandler(handler)
            if logger_name not in self._bridged_logger_names:
                self._bridged_logger_names.append(logger_name)

    def close(self):
        """
        Closes all handlers associated with the logger.
        """
        for logger_name in self._bridged_logger_names:
            external_logger = logging.getLogger(logger_name)
            for handler in self.handlers:
                external_logger.removeHandler(handler)
        self._bridged_logger_names.clear()
        for handler in self.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        self.handlers.clear()
        logging.shutdown()