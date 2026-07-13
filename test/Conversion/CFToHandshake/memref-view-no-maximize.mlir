// RUN: circt-opt -lower-cf-to-handshake %s | FileCheck %s

// A memref VALUE (here a memref.reinterpret_cast view) that is live across a
// loop must NOT be threaded through block arguments by SSA maximization.
// insertMergeOps deliberately skips memref block arguments (memories are handled
// separately), so a maximized memref block argument would survive into
// removeBlockOperands and abort with "Cannot destroy a value that still has
// uses". The SSA-maximization strategy must exclude any op that produces a
// memref result (not just allocations), so the view stays a direct dominating
// reference and no memref block argument is created.

// It must lower cleanly (no crash): the view becomes a memory op and no memref
// block argument is created on the loop blocks.
// CHECK-LABEL:   handshake.func @memview_loop(
// CHECK:           memory{{.*}}memref<64xi32, strided<[1]>>
// CHECK:           return
func.func @memview_loop(%arg0: memref<128xi32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %view = memref.reinterpret_cast %arg0 to offset: [0], sizes: [64], strides: [1]
      : memref<128xi32> to memref<64xi32, strided<[1]>>
  cf.br ^bb1(%c0 : index)
^bb1(%i: index):
  %cond = arith.cmpi slt, %i, %n : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %v = memref.load %view[%i] : memref<64xi32, strided<[1]>>
  memref.store %v, %view[%i] : memref<64xi32, strided<[1]>>
  %ni = arith.addi %i, %c1 : index
  cf.br ^bb1(%ni : index)
^bb3:
  return
}
