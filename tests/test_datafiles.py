import pytest
# import os, sys
import numpy as np
import numpy.testing as npt

from pykima.utils import read_datafile


def test_skip(tmpdir):
    f = tmpdir.join('data.txt')
    f.write('c1  c2  c3\n1.0  2.0  3.0')

    with pytest.raises(ValueError):  # wrong skip value
        read_datafile(f, skip=0)

    with pytest.raises(IndexError):  # file has no 4th column
        read_datafile(f, skip=1)

    f.write('c1  c2  c3  c4\n1.0  2.0  3.0  1')
    data, obs = read_datafile(f, skip=1)

    with pytest.raises(ValueError):
        with pytest.warns(UserWarning):
            read_datafile(f, skip=2)


def test_obs(tmpdir):
    f = tmpdir.join('data.txt')
    f.write(
        """jdb\tvrad\tsvrad\tinst
        ---\t----\t-----\t----
        1.0\t2.0\t3.0\t1
        1.1\t2.1\t3.1\t2
        1.2\t2.2\t3.2\t1
        1.3\t2.3\t3.3\t1
    """)

    data, obs = read_datafile(f, skip=2)

    data_should_be = (np.arange(1, 1.3, 0.1) + [[0], [1], [2]]).T
    npt.assert_allclose(data, data_should_be)

    npt.assert_array_equal(obs, [1, 2, 1, 1])


def test_dtypes(tmpdir):
    f = tmpdir.join('data.txt')
    f.write('c1  c2  c3  c4\n1.0  2.0  3.0  1')

    data, obs = read_datafile(f, skip=1)

    npt.assert_array_equal(data, np.array([1, 2, 3]))
    npt.assert_array_equal(obs, np.array([1]))
    assert data.dtype == float
    assert obs.dtype == int
