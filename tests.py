import unittest
from unittest.mock import patch
import numpy as np
import modeling

class TestReadData(unittest.TestCase):
    @patch('modeling.da.readDataAll_p')
    @patch('modeling.np.random.permutation')
    @patch('modeling.da.categorize_columns')
    @patch('modeling.ic.apply_pca_to_categories')
    @patch('modeling.ic.apply_ica_to_categories')
    def test_read_data(self, mock_apply_ica, mock_apply_pca, mock_categorize, mock_permutation, mock_readDataAll_p):
    # Setup mock returns
    mock_readDataAll_p.return_value = ('data', 'annotations')
    mock_permutation.return_value = np.array([0, 1, 2])
    mock_categorize.return_value = {'mocked_data': 'mocked_value'}
    mock_apply_pca.return_value = 'component_data_pca'
    mock_apply_ica.return_value = 'component_data_ica'

    # Test without component reduction and without shuffling
    data, annotations = modeling.read_data(apply_comp_reduction=False, shuffle=False)
    self.assertEqual(data, 'data')
    self.assertEqual(annotations, 'annotations')

    # Test without component reduction but with shuffling
    data, annotations = modeling.read_data(apply_comp_reduction=False, shuffle=True)
    mock_permutation.assert_called_once()

    # Test with PCA component reduction
    modeling.DIM_REDUCTION_MODEL = modeling.DIM_REDUCTION.PCA
    data, annotations = modeling.read_data(apply_comp_reduction=True)
    mock_apply_pca.assert_called_once()

    # Test with ICA component reduction
    modeling.DIM_REDUCTION_MODEL = modeling.DIM_REDUCTION.ICA
    data, annotations = modeling.read_data(apply_comp_reduction=True)
    mock_apply_ica.assert_called_once()

    # Test with invalid dimensionality reduction model
    modeling.DIM_REDUCTION_MODEL = 'INVALID'
    with self.assertRaises(SystemExit):
        modeling.read_data(apply_comp_reduction=True)

if __name__ == '__main__':
    unittest.main()