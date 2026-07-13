// RUN: circt-opt -handshake-lower-extmem-to-hw %s | FileCheck %s

// Regression: a function with TWO external memories where the FIRST is INOUT
// (ld=1, st=1 -> inserts a load AND a store port, then erases the memref: a net
// +1 to the argument list). The second memref's ports must be inserted, and its
// argument erased, at its LIVE index -- not its original index. Using the stale
// original index inserts the second memory's ports in the wrong place and erases
// a still-used store-port argument, which previously aborted with
// "Cannot destroy a value that still has uses". This checks it lowers cleanly and
// both memories are converted to address/data struct outputs.

// CHECK-LABEL:   handshake.func @two_mem_first_inout(
// CHECK-NOT:       memref
// CHECK:           return
handshake.func @two_mem_first_inout(%addr0: index, %addr1: index, %v: i32,
                                    %m0: memref<8xi32>, %m1: memref<8xi32>,
                                    %ctrl: none) -> none {
  %ld0, %stC0, %ldC0 = handshake.extmemory[ld=1, st=1](%m0 : memref<8xi32>)(%sd0, %sa0, %la0) {id = 0 : i32} : (i32, index, index) -> (i32, none, none)
  %ld1, %ldC1 = handshake.extmemory[ld=1, st=0](%m1 : memref<8xi32>)(%la1) {id = 1 : i32} : (index) -> (i32, none)
  %f:4 = fork [4] %ctrl : none
  %ldd0, %la0 = load [%addr0] %ld0, %f#0 : index, i32
  %sd0, %sa0 = store [%addr1] %v, %f#1 : index, i32
  %ldd1, %la1 = load [%addr0] %ld1, %f#2 : index, i32
  sink %ldd0 : i32
  sink %ldd1 : i32
  %j = join %stC0, %ldC0, %ldC1, %f#3 : none, none, none, none
  return %j : none
}
