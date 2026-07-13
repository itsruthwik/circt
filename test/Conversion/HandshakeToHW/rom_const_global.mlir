// RUN: circt-opt -lower-handshake-to-hw -split-input-file %s | FileCheck %s

// A load-only handshake.memory carrying a captured constant-global initializer
// (attribute "vtr.rom_init", attached by the CFToHandshake memory lowering) must
// lower to a combinational array-mux ROM (hw.array_create + hw.array_get) with
// the initial values baked in -- NOT to an uninitialized seq.hlmem.

// CHECK-LABEL: hw.module @handshake_memory
// CHECK-NOT:     seq.hlmem
// CHECK-DAG:     hw.constant 10 : i32
// CHECK-DAG:     hw.constant 20 : i32
// CHECK-DAG:     hw.constant 30 : i32
// CHECK-DAG:     hw.constant 40 : i32
// CHECK:         hw.array_create
// CHECK:         hw.array_get
handshake.func @rom(%ldaddr: index, %ctrl: none) -> (i32, none) {
  sink %ctrl : none
  %0:2 = memory[ld = 1, st = 0] (%ldaddr) {id = 0 : i32, lsq = false, vtr.rom_init = dense<[10, 20, 30, 40]> : tensor<4xi32>} : memref<4xi32>, (index) -> (i32, none)
  return %0#0, %0#1 : i32, none
}
