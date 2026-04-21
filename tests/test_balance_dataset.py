import sys
import os
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_dependencies():
    """Fixture to mock missing libraries and clean up after tests."""
    # Define the modules to mock
    mock_modules = [
        "numpy",
        "skimage",
        "skimage.transform",
        "imblearn",
        "imblearn.over_sampling"
    ]

    # Store original modules if they exist
    original_modules = {name: sys.modules.get(name) for name in mock_modules}

    # Create and set mocks
    mocks = {name: MagicMock() for name in mock_modules}
    for name, m in mocks.items():
        sys.modules[name] = m

    # Custom mock for numpy.unique to handle multiple return values
    def mock_unique(y, return_counts=False):
        # Convert y to list for easy processing
        if hasattr(y, 'tolist'):
            y_list = y.tolist()
        else:
            y_list = list(y)

        classes = sorted(list(set(y_list)))

        # Mock class array
        classes_arr = MagicMock()
        classes_arr.__len__.return_value = len(classes)
        classes_arr.tolist.return_value = classes

        if return_counts:
            counts = [y_list.count(c) for c in classes]
            counts_arr = MagicMock()
            counts_arr.tolist.return_value = counts
            return classes_arr, counts_arr
        return classes_arr

    mocks["numpy"].unique = mock_unique

    # Mock numpy.empty and numpy.zeros to return MagicMocks
    mocks["numpy"].empty.return_value = MagicMock()
    mocks["numpy"].zeros.return_value = MagicMock()

    yield mocks

    # Restore original modules or remove mocks
    for name, original in original_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

def test_balance_dataset_success(mock_dependencies):
    """Test successful dataset balancing."""
    # Ensure balance_dataset is freshly imported or reloaded
    if 'scripts.balance_dataset' in sys.modules:
        del sys.modules['scripts.balance_dataset']
    from scripts.balance_dataset import balance_dataset

    # Mock data
    mock_images = MagicMock()
    mock_images.shape = (10, 128, 128, 3)
    mock_images.__len__.return_value = 10

    mock_labels = MagicMock()
    mock_labels.tolist.return_value = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    mock_data = MagicMock()
    mock_data.__enter__.return_value = {'images': mock_images, 'labels': mock_labels}

    with patch('numpy.load', return_value=mock_data),          patch('numpy.savez_compressed') as mock_save,          patch('scripts.balance_dataset.RandomOverSampler') as mock_ros,          patch('scripts.balance_dataset.resize', side_effect=lambda x, size, **kwargs: MagicMock()),          patch('scripts.balance_dataset.BASE_DIR', '/tmp'),          patch('logging.info'),          patch('logging.error') as mock_log_error:

        # Setup ROS mock
        mock_ros_instance = mock_ros.return_value
        mock_ros_instance.fit_resample.return_value = (MagicMock(), mock_labels)

        result = balance_dataset()

        assert result is True
        assert mock_save.called
        mock_log_error.assert_not_called()

def test_balance_dataset_non_binary(mock_dependencies):
    """Test handling of non-binary classification data."""
    if 'scripts.balance_dataset' in sys.modules:
        del sys.modules['scripts.balance_dataset']
    from scripts.balance_dataset import balance_dataset

    # Mock data with 3 classes
    mock_images = MagicMock()
    mock_images.shape = (10, 128, 128, 3)

    mock_labels = MagicMock()
    mock_labels.tolist.return_value = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]

    mock_data = MagicMock()
    mock_data.__enter__.return_value = {'images': mock_images, 'labels': mock_labels}

    with patch('numpy.load', return_value=mock_data),          patch('scripts.balance_dataset.BASE_DIR', '/tmp'),          patch('logging.info'),          patch('logging.error') as mock_log_error:

        result = balance_dataset()

        assert result is False
        mock_log_error.assert_called()
        args, _ = mock_log_error.call_args
        assert "Expected binary classification data" in args[0]

def test_balance_dataset_exception(mock_dependencies):
    """Test handling of general exceptions during balancing."""
    if 'scripts.balance_dataset' in sys.modules:
        del sys.modules['scripts.balance_dataset']
    from scripts.balance_dataset import balance_dataset

    with patch('numpy.load', side_effect=Exception("File not found")),          patch('scripts.balance_dataset.BASE_DIR', '/tmp'),          patch('logging.info'),          patch('logging.error') as mock_log_error:

        result = balance_dataset()

        assert result is False
        mock_log_error.assert_called()
        args, _ = mock_log_error.call_args
        assert "Balancing failed: File not found" in args[0]
