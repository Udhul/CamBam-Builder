"""Exact supported matrix boundary: XY affine plus independent Z translation."""
import unittest

import numpy as np

from cambam_builder.cad_transformations import (
    from_cambam_matrix_str, from_cambam_matrix_str_v2,
    to_cambam_matrix_str, to_cambam_matrix_str_v2,
)


class ZMatrixTests(unittest.TestCase):
    def test_affine_xy_and_z_translation_have_independent_round_trip(self):
        xy = np.array([[0.0, -2.0, 4.0], [3.0, 0.5, -7.0], [0.0, 0.0, 1.0]])
        for z in (0.0, 2.75, -9.125):
            for encoder in (to_cambam_matrix_str, to_cambam_matrix_str_v2):
                text = encoder(xy, z_offset=z)
                self.assertEqual(float(text.split()[14]), z)
                for decoder in (from_cambam_matrix_str, from_cambam_matrix_str_v2):
                    restored, offset = decoder(text, return_z=True)
                    np.testing.assert_array_equal(restored, xy)
                    self.assertEqual(offset, z)
                    if z:
                        with self.assertRaisesRegex(ValueError, "refusing to discard"):
                            decoder(text)

    def test_unsupported_spatial_and_perspective_components_are_rejected(self):
        for row, col, value in ((0, 2, 1e-14), (1, 2, 1), (2, 0, 1),
                                (2, 1, -1), (2, 2, 2), (3, 0, 1),
                                (3, 1, 1), (3, 2, 1), (3, 3, 2)):
            matrix = np.eye(4)
            matrix[row, col] = value
            text = " ".join(map(str, matrix.flatten(order="F")))
            for decoder in (from_cambam_matrix_str, from_cambam_matrix_str_v2):
                with self.subTest(row=row, col=col, decoder=decoder.__name__):
                    with self.assertRaises(ValueError):
                        decoder(text, return_z=True)

    def test_nonfinite_and_malformed_matrices_fail(self):
        for text in ("", "1 2", " ".join(["nan"] * 16), " ".join(["inf"] * 16)):
            with self.assertRaises(ValueError):
                from_cambam_matrix_str(text, return_z=True)
        for matrix, z in ((np.eye(4), 0), (np.eye(3), float("nan")),
                          (np.full((3, 3), float("inf")), 0)):
            with self.assertRaises(ValueError):
                to_cambam_matrix_str(matrix, z_offset=z)
        xy, z = from_cambam_matrix_str(" Identity ", return_z=True)
        np.testing.assert_array_equal(xy, np.eye(3))
        self.assertEqual(z, 0.0)


if __name__ == "__main__":
    unittest.main()
