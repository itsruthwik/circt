// RUN: circt-opt -lower-cf-to-handshake %s | FileCheck %s

// A data-dependent NESTED loop. A handshake loop needs an initial control token
// (a priming register) to start circulating; priming only the outermost loop
// leaves the inner loop's recurrence un-primed, so it never starts and the
// circuit deadlocks. lower-cf-to-handshake primes EVERY loop
// (getLoopsInPreorder), not just the top-level ones, so this lowers to TWO
// priming registers -- one for the outer loop and one for the inner loop.

// CHECK-LABEL:   handshake.func @nested_loops
// CHECK-COUNT-2: initValues
// CHECK:         return
func.func @nested_loops(%arg0: i32) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  cf.br ^bb1(%c0_i32 : i32)
^bb1(%i: i32):  // 2 preds: ^bb0, ^bb4  (outer loop header)
  %co = arith.cmpi slt, %i, %arg0 : i32
  cf.cond_br %co, ^bb2(%c0_i32 : i32), ^bb5(%c0_i32 : i32)
^bb2(%j: i32):  // 2 preds: ^bb1, ^bb3  (inner loop header)
  %ci = arith.cmpi slt, %j, %i : i32
  cf.cond_br %ci, ^bb3(%j : i32), ^bb4
^bb3(%jj: i32):  // pred: ^bb2  (inner body)
  %jn = arith.addi %jj, %c1_i32 : i32
  cf.br ^bb2(%jn : i32)
^bb4:  // pred: ^bb2  (outer body / inner exit)
  %in = arith.addi %i, %c1_i32 : i32
  cf.br ^bb1(%in : i32)
^bb5(%r: i32):  // pred: ^bb1  (exit)
  return %r : i32
}
