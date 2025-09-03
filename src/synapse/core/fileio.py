from abc import ABC, abstractmethod
import logging
import math
import os
from pathlib import Path

import awkward as ak
import uproot

_logger = logging.getLogger("SynapseLogger")

class FileReaderBase(ABC):
    """
    Abstract base class for file reading, defines a unified file reading interface
    """

    def __init__(
            self,
            file_path: str,
            keys: list[str],
            load_range: tuple[float, float] | tuple[int, int] = (0.0, 1.0),
            normalized_range: bool = True
    ):
        """
        Initialize the file reader

        Args:
            file_path (str): file path
            keys (list[str]): list of keys to read
            load_range (tuple[float, float] | tuple[int, int]):
                range of data to load, elements' data type should be `float`
                when normalized_range is True, `int` otherwise
            normalized_range (bool): whether the `load_range` is normalized
        """
        # Validate load_range
        if not isinstance(load_range, tuple) or len(load_range) != 2:
            raise TypeError("load_range must be a tuple with exactly 2 elements")

        # Check if load_range values are in ascending order
        if load_range[0] >= load_range[1]:
            raise ValueError("load_range must have lower bound less than upper bound")

        entries = self.get_entries()
        # TODO: truncate load_range to proper range instead of raising error
        if normalized_range:
            if not all(isinstance(x, float) for x in load_range):
                raise TypeError("When normalized_range is True, load_range elements must be float")
            # Check if normalized range is within [0.0, 1.0]
            if load_range[0] < 0.0 or load_range[1] > 1.0:
                raise ValueError("Normalized load_range must be within [0.0, 1.0]")
            self.load_range = (math.trunc(load_range[0] * entries), math.trunc(load_range[1] * entries))
        else:
            if not all(isinstance(x, int) for x in load_range):
                raise TypeError("When normalized_range is False, load_range elements must be int")
            # Check non-normalized range bounds
            if entries > 0:
                if load_range[0] < 0 or load_range[1] > entries:
                    raise ValueError(f"Non-normalized load_range must be within [0, {entries}]")
            self.load_range = load_range

        self.file_path = file_path
        self.keys = keys
        self.normalized_range = normalized_range
        self.file = None
        self.data = None

    @abstractmethod
    def read(self):
        """
        Abstract method for reading files, subclasses need to implement specific reading logic

        Returns:
            specific return format is determined by subclasses
        """
        pass

    @abstractmethod
    def get_entries(self) -> int:
        """
        Get the number of entries in the file

        Returns:
            int: number of entries
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close file resources
        """
        pass

    def __enter__(self):
        """
        Support using with statement for context management
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Automatically close resources when the with statement block ends
        """
        self.close()


class ROOTFileReader(FileReaderBase):
    """
    Class for reading ROOT files using uproot library
    """

    def __init__(
            self,
            file_path: str,
            branches: list[str],
            tree_name: str = None,
            load_range: tuple[float, float] | tuple[int, int] = (0.0, 1.0),
            normalized_range: bool = True
    ):
        """
        Initialize the ROOT file reader

        Args:
            file_path (str): path to the ROOT file
            branches (list[str]): list of branches to read
            tree_name (str, optional): name of the TTree to read.
            load_range (tuple[float, float] | tuple[int, int]):
                range of data to load, elements' data type should be `float`
                when normalized_range is True, `int` otherwise
            normalized_range (bool): whether the `load_range` is normalized
        """
        self.tree_name = tree_name
        self.tree = None

        self._entries = self._init_file(file_path)

        super().__init__(file_path, branches, load_range, normalized_range)

    def _init_file(self, file_path: str) -> int:
        """
        Initialize the ROOT file and get entries count

        Args:
            file_path (str): path to the ROOT file

        Returns:
            int: number of entries in the file
        """
        try:
            self.file = uproot.open(file_path)

            # If tree_name is not provided, try to get one
            if self.tree_name is None:
                tree_names = self._get_tree_names(self.file)
                if len(tree_names) == 1:
                    self.tree_name = tree_names.pop()
                    _logger.info(f"No tree_name specified, using tree '{self.tree_name}' from file: \n{file_path}")
                elif len(tree_names) == 0:
                    raise ValueError(f"No TTrees found in file {file_path}")
                else:
                    raise ValueError(f"Multiple TTrees found in file {file_path}. Please specify tree_name.")

            # Get the tree
            self.tree = self.file[self.tree_name]

            # Return the number of entries
            return self.tree.num_entries

        except Exception as e:
            raise RuntimeError(f"Failed to open ROOT file {file_path}: {str(e)}")

    def _get_tree_names(self, directory, current_path: str = "") -> set:
        """
        Get the names of all trees in the file

        Returns:
            set: Set of tree names
        """
        tree_names = set()
        for key in directory.keys():
            full_path = f"{current_path}/{key}" if current_path else key
            try:
                obj = directory[key]
                if isinstance(obj, uproot.TTree):
                    tree_names.add(full_path)
                elif isinstance(obj, uproot.ReadOnlyDirectory):
                    self._get_tree_names(obj, full_path)
            except Exception as e:
                _logger.debug(f"Unable to process {full_path}: {str(e)}")
                continue
        return tree_names

    def read(self):
        """
        Read the data from the ROOT file

        Returns:
            dict: Dictionary containing branch arrays
        """
        if self.tree is None:
            raise RuntimeError("Tree not initialized")

        start, stop = self.load_range
        try:
            # Read the data as awkward arrays
            self.data = self.tree.arrays(filter_name=self.keys, entry_start=start, entry_stop=stop)
        except Exception as e:
            raise RuntimeError(f"Failed to read data from ROOT file: {str(e)}")

    def get_entries(self) -> int:
        """
        Get the number of entries in the file

        Returns:
            int: number of entries
        """
        return self._entries

    def get_data(self) -> ak.Array:
        """
        Get the read data, read if not already done

        Returns:
            dict: Dictionary containing branch arrays
        """
        if self.data is None:
            self.read()
        return self.data

    def close(self):
        """
        Close the file resources
        """
        if self.file is not None:
            self.file.close()
            self.file = None
            self.tree = None
            self.data = None


# TODO: implement HDF5 file reader

def read_files(
        file_paths: list[str],
        keys: list[str],
        load_range: tuple = (0.0, 1.0),
        normalized_range: bool = True,
        merge: bool = True,
        **kwargs
):
    """
    Read multiple files and return the data

    Args:
        file_paths (list[str]): list of file paths
        keys (list[str]): list of keys to read
        load_range (tuple): range of data to load
        normalized_range (bool): whether the `load_range` is normalized
        merge (bool): whether to merge the data from all files
        **kwargs: additional arguments for the file reader

    Returns:
        ak.Array / list[ak.Array]: Concatenated / list of data from all files
        list[str]: list of file names
    """
    data = []
    file_names = []
    for file_path in file_paths:
        # Store file names
        file_names.append(Path(file_path).name)
        # Get file extension
        file_extension = Path(file_path).suffix.lstrip(".")

        # TODO: Take FileReader class as an argument, let the caller decide which reader to use
        # Select appropriate reader based on file extension
        if file_extension == 'root':
            reader = ROOTFileReader(file_path,
                                    branches=keys,
                                    tree_name=kwargs.get("tree_name", None),
                                    load_range=load_range,
                                    normalized_range=normalized_range)
        else:
            raise RuntimeError(f"File: {file_path} has unsupported file extension: {file_extension}")

        # Read the data
        try:
            file_data = reader.get_data()
            if file_data is None:
                _logger.warning(f"No data read from file {file_path}")
            else:
                data.append(file_data)
        except Exception as e:
            _logger.error(f"Error reading file {file_path}: {str(e)}")
        finally:
            reader.close()

    if len(data) != 0:
        if merge:
            # concatenate the data from all files
            # TODO: after implementing the reader for other file formats,
            #       we need to check if the data is of the same type
            #       we may need to define an adaptive concatenation function
            data = ak.concatenate(data)
    else:
        raise RuntimeError("No data read from the provided files: \n" + ", \n".join(file_names))

    return data, file_names


class FileWriterBase(ABC):
    """
    Abstract base class for file writing, defines a unified file writing interface
    """

    def __init__(
            self,
            file_path: str,
            data
    ):
        """
        Initialize the file writer

        Args:
            file_path (str): file path
            data: data to write
        """
        self.file_path = file_path
        self.data = data

    @abstractmethod
    def write(self):
        """
        Abstract method for writing files, subclasses need to implement specific writing logic

        Returns:
            specific return format is determined by subclasses
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close file resources
        """
        pass

    def __enter__(self):
        """
        Support using with statement for context management
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Automatically close resources when the with statement block ends
        """
        self.close()


class ROOTFileWriter(FileWriterBase):
    """
    Class for writing ROOT files using uproot library with chunked writing support
    """

    def __init__(
            self,
            file_path: str,
            data: ak.Array,
            tree_name: str = "tree",
            compression: uproot.compression.Compression = uproot.LZ4(4),
            chunk_size: int = 10000
    ):
        """
        Initialize the ROOT file writer with chunked writing support

        Args:
            file_path (str): Path to the ROOT file
            data (ak.Array): Data to write (Awkward Array)
            tree_name (str, optional): Name of the TTree. Defaults to "tree".
            compression (uproot.compression.Compression, optional): Compression settings.
                Defaults to uproot.LZ4(4).
            chunk_size (int, optional): Number of entries per write chunk.
                Defaults to 10000.
        """
        super().__init__(file_path, data)
        self.tree_name = tree_name
        self.compression = compression
        self.chunk_size = chunk_size
        self._file_handle = None

        # Precompute metadata
        self._fields = list(data.fields) if data.fields else []
        self._schema = {k: data[k].type for k in self._fields}
        self._total_entries = len(data[self._fields[0]]) if self._fields else 0

        # TODO: Automatically determine chunk size based on data size and available memory

    def write(self) -> None:
        """
        Write data to ROOT file with chunking mechanism
        """
        if self._total_entries == 0:
            raise ValueError("Cannot write empty array")

        # Initialize file handle
        self._file_handle = uproot.recreate(
            self.file_path,
            compression=self.compression
        )
        tree = self._file_handle.mktree(self.tree_name, self._schema)

        # Chunked writing
        start = 0
        while start < self._total_entries:
            end = min(start + self.chunk_size, self._total_entries)
            chunk = {k: self.data[k][start:end] for k in self._fields}
            tree.extend(chunk)
            start = end

    def close(self) -> None:
        """
        Explicitly close file resources
        """
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()


# TODO: implement HDF5 file writer

def write_file(
        file_path: str,
        data: ak.Array,
        **kwargs
) -> None:
    """
    Write data to a file

    Args:
        file_path (str): path to the file
        data (ak.Array): data to write
        **kwargs: additional arguments for the file writer

    Returns:
    """
    # Check if the file path is valid
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Select appropriate writer based on file extension
    file_extension = Path(file_path).suffix.lstrip(".")
    if file_extension == 'root':
        writer = ROOTFileWriter(file_path,
                                data,
                                tree_name=kwargs.get("tree_name", "tree"),
                                compression=kwargs.get("compression", uproot.LZ4(4)),
                                chunk_size=kwargs.get("chunk_size", 10000))
    else:
        raise RuntimeError(f"File: {file_path} has unsupported file extension: {file_extension}")

    # Write the data
    try:
        writer.write()
    except Exception as e:
        raise RuntimeError(f"Failed to write data to file {file_path}: {str(e)}")
    finally:
        writer.close()