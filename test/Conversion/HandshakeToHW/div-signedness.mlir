// RUN: circt-opt --lower-handshake-to-hw %s | FileCheck %s

// Regression: arith.divui/divsi must lower to comb.divu/divs respectively (the
// UnitRateConversionPattern pair was swapped, so unsigned division emitted a signed
// divider and vice versa -- a silent miscompile for any dividend with the high bit set).

// CHECK-LABEL: hw.module @div_signedness
// CHECK-DAG: comb.divu
// CHECK-DAG: comb.divs
handshake.func @div_signedness(%a: i32, %b: i32, %c: i32, %d: i32, %ctrl: none) -> (i32, i32, none) {
  %u = arith.divui %a, %b : i32
  %s = arith.divsi %c, %d : i32
  return %u, %s, %ctrl : i32, i32, none
}
