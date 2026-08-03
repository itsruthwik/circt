// RUN: circt-opt -split-input-file -hw-legalize-modules -verify-diagnostics %s | FileCheck %s

// Every struct here uses fields of *different* widths on purpose: with equal
// widths a reversed field order would produce identical-looking IR, and the
// `from <N>` offsets below are the only thing pinning the layout.

module attributes {circt.loweringOptions = "disallowPackedStructs"} {

// The shape produced by hw-flatten-io when it flattens a struct-typed port: a
// bitcast from the flat vector, exploded back into fields.  The first field
// occupies the MSBs, so `address` is the high 5 bits of the 37.

// CHECK-LABEL: hw.module @bitcast_explode
hw.module @bitcast_explode(in %flat : i37, out address : i5, out data : i32) {
  %s = hw.bitcast %flat : (i37) -> !hw.struct<address: i5, data: i32>
  %address, %data = hw.struct_explode %s : !hw.struct<address: i5, data: i32>
  hw.output %address, %data : i5, i32
}
// CHECK-DAG:   %[[ADDR:.+]] = comb.extract %flat from 32 : (i37) -> i5
// CHECK-DAG:   %[[DATA:.+]] = comb.extract %flat from 0 : (i37) -> i32
// CHECK:       hw.output %[[ADDR]], %[[DATA]]
// CHECK-NOT:   hw.struct

}

// -----

module attributes {circt.loweringOptions = "disallowPackedStructs"} {

// Three fields of distinct widths: exercises the running offset accumulation,
// which two fields cannot distinguish from "first field high, second low".
// Widths 3/7/5 over i15 give offsets 12, 5 and 0.

// CHECK-LABEL: hw.module @three_fields
hw.module @three_fields(in %flat : i15, out a : i3, out b : i7, out c : i5) {
  %s = hw.bitcast %flat : (i15) -> !hw.struct<a: i3, b: i7, c: i5>
  %a, %b, %c = hw.struct_explode %s : !hw.struct<a: i3, b: i7, c: i5>
  hw.output %a, %b, %c : i3, i7, i5
}
// CHECK-DAG:   %[[A:.+]] = comb.extract %flat from 12 : (i15) -> i3
// CHECK-DAG:   %[[B:.+]] = comb.extract %flat from 5 : (i15) -> i7
// CHECK-DAG:   %[[C:.+]] = comb.extract %flat from 0 : (i15) -> i5
// CHECK:       hw.output %[[A]], %[[B]], %[[C]]
// CHECK-NOT:   hw.struct

}

// -----

module attributes {circt.loweringOptions = "disallowPackedStructs"} {

// A single-field consumer rather than a full explode.

// CHECK-LABEL: hw.module @struct_extract_user
hw.module @struct_extract_user(in %flat : i37, out data : i32) {
  %s = hw.bitcast %flat : (i37) -> !hw.struct<address: i5, data: i32>
  %data = hw.struct_extract %s["data"] : !hw.struct<address: i5, data: i32>
  hw.output %data : i32
}
// CHECK:       %[[DATA:.+]] = comb.extract %flat from 0 : (i37) -> i32
// CHECK:       hw.output %[[DATA]]
// CHECK-NOT:   hw.struct

}

// -----

module attributes {circt.loweringOptions = "disallowPackedStructs"} {

// hw.struct_create's operands are already the fields, so no slicing is needed.

// CHECK-LABEL: hw.module @struct_create
hw.module @struct_create(in %a : i5, in %b : i32, out data : i32) {
  %s = hw.struct_create (%a, %b) : !hw.struct<address: i5, data: i32>
  %data = hw.struct_extract %s["data"] : !hw.struct<address: i5, data: i32>
  hw.output %data : i32
}
// CHECK:       hw.output %b
// CHECK-NOT:   hw.struct

}

// -----

module attributes {circt.loweringOptions = "disallowPackedStructs"} {

// A struct that reaches an op we cannot rewrite must be reported, not emitted.

hw.module @reject_unsupported_user(in %flat : i37, in %sel : i1,
                                   out out : !hw.struct<address: i5, data: i32>) {
  %s = hw.bitcast %flat : (i37) -> !hw.struct<address: i5, data: i32>
  // expected-error @+1 {{unsupported packed struct expression}}
  %m = comb.mux %sel, %s, %s : !hw.struct<address: i5, data: i32>
  hw.output %m : !hw.struct<address: i5, data: i32>
}

}

// -----

module attributes {circt.loweringOptions = "disallowPackedStructs"} {

// A nested aggregate field has no single bit slice; hard-error rather than
// mis-slicing it.

hw.module @reject_nested_aggregate(in %flat : i10, out out : !hw.array<2xi3>) {
  // expected-error @+1 {{unsupported packed struct expression}}
  %s = hw.bitcast %flat : (i10) -> !hw.struct<a: i4, b: !hw.array<2xi3>>
  %a, %b = hw.struct_explode %s : !hw.struct<a: i4, b: !hw.array<2xi3>>
  hw.output %b : !hw.array<2xi3>
}

}

// -----

// Without the option the struct is left alone: the lowering is opt-in, so
// targets that accept `struct packed` are unaffected.

// CHECK-LABEL: hw.module @option_off
hw.module @option_off(in %flat : i37, out address : i5, out data : i32) {
  %s = hw.bitcast %flat : (i37) -> !hw.struct<address: i5, data: i32>
  %address, %data = hw.struct_explode %s : !hw.struct<address: i5, data: i32>
  hw.output %address, %data : i5, i32
}
// CHECK:       hw.bitcast
// CHECK:       hw.struct_explode
