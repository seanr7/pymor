# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.operators.block import BlockDiagonalOperator, BlockOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test_apply():
    A11 = np.random.randn(2, 3)
    A12 = np.random.randn(2, 4)
    A21 = np.zeros((5, 3))
    A22 = np.random.randn(5, 4)
    A = np.vstack((np.hstack((A11, A12)),
                   np.hstack((A21, A22))))
    A11op = NumpyMatrixOperator(A11)
    A12op = NumpyMatrixOperator(A12)
    A22op = NumpyMatrixOperator(A22)
    Aop = BlockOperator(np.array([[A11op, A12op], [None, A22op]]))

    v1 = np.random.randn(3)
    v2 = np.random.randn(4)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Aop.apply(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(A.dot(v), w)


def test_apply_adjoint():
    A11 = np.random.randn(2, 3)
    A12 = np.random.randn(2, 4)
    A21 = np.zeros((5, 3))
    A22 = np.random.randn(5, 4)
    A = np.vstack((np.hstack((A11, A12)),
                   np.hstack((A21, A22))))
    A11op = NumpyMatrixOperator(A11)
    A12op = NumpyMatrixOperator(A12)
    A22op = NumpyMatrixOperator(A22)
    Aop = BlockOperator(np.array([[A11op, A12op], [None, A22op]]))

    v1 = np.random.randn(2)
    v2 = np.random.randn(5)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Aop.apply_adjoint(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(A.T.dot(v), w)


def test_block_diagonal():
    A = np.random.randn(2, 3)
    B = np.random.randn(4, 5)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))
    assert Cop.source.dim == 8
    assert Cop.range.dim == 6


def test_blk_diag_apply_inverse():
    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Cop.apply_inverse(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(spla.solve(C, v), w)


def test_blk_diag_apply_inverse_adjoint():
    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Cop.apply_inverse_adjoint(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(spla.solve(C.T, v), w)


def test_sparse():
    op = np.empty((4, 4), dtype=object)
    for I in range(3):
        op[I, I] = NumpyMatrixOperator(np.eye(20))
    op[3, 3] = NumpyMatrixOperator(np.eye(10))
    op[0, 2] = NumpyMatrixOperator(4 * np.eye(20))
    block_op = BlockOperator(op, make_sparse=True)
    ones = block_op.range.ones()
    block_op.apply(ones)
    block_op.apply_adjoint(ones)
    block_op.assemble()
    block_op.d_mu('something')
    dense_block_op = block_op.to_dense()
    dense_block_op.assemble()
    a = dense_block_op.as_source_array().to_numpy()
    b = block_op.as_source_array().to_numpy()
    assert np.allclose(a, b)
    a = dense_block_op.as_range_array().to_numpy()
    b = block_op.as_range_array().to_numpy()
    assert np.allclose(a, b)
    assert block_op.apply2(ones, ones) == dense_block_op.apply2(ones, ones)
