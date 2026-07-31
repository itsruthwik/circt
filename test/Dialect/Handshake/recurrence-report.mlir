// RUN: circt-opt -handshake-recurrence-report %s -o /dev/null | FileCheck %s
// RUN: circt-opt -handshake-recurrence-report=format=json %s -o /dev/null | FileCheck %s -check-prefix=JSON
// RUN: not circt-opt -handshake-recurrence-report=format=bogus %s -o /dev/null 2>&1 | FileCheck %s -check-prefix=BADFORMAT

// BADFORMAT: unknown recurrence report format: bogus

// A loop with two sequential buffer slots on its index recurrence. The buffer
// outside the cycle carries four slots and must not count towards the II.
//
// The cycle carries two tokens -- one from the priming register's initValues
// and one from the mux arm entering from outside the loop -- so two slots over
// two tokens gives a recurrence bound of 1.

// CHECK-LABEL: handshake.func @seqRecurrence: 2 recurrence(s)
// CHECK:         recurrence 0: 6 ops, 2 seq buffer(s) (2 slot(s)), 0 fifo buffer(s) (0 slot(s)), 1 init token(s)
// CHECK:         II: 1.00  |  if fifo were transparent: 1.00
// CHECK:         slots over whole component: 2 (as lowered), 2 (if fifo were transparent)

// JSON: "function": "seqRecurrence"
// JSON: "ii": 1
// JSON: "ii-if-fifo-transparent": 1
handshake.func @seqRecurrence(%arg0: none, ...) -> (i32, none) {
  %limit = constant %arg0 {value = 100 : i32} : i32
  %zero = constant %arg0 {value = 0 : i32} : i32
  %one = constant %arg0 {value = 1 : i32} : i32
  // Outside the recurrence: must not contribute to the II.
  %offCycle = buffer [4] seq %limit : i32
  %prime = buffer [1] seq %cond {initValues = [0]} : i1
  %idx = mux %prime [%zero, %next] : i1, i32
  %ctrl = mux %prime [%arg0, %trueCtrl] : i1, none
  %idxBuf = buffer [1] seq %idx : i32
  %cond = arith.cmpi slt, %idxBuf, %offCycle : i32
  %trueIdx, %falseIdx = cond_br %cond, %idxBuf : i32
  %trueCtrl, %falseCtrl = cond_br %cond, %ctrl : none
  %next = arith.addi %trueIdx, %one : i32
  return %falseIdx, %falseCtrl : i32, none
}

// The control recurrence carries no buffer at all: a combinational loop, which
// the report surfaces as a zero initiation interval.

// CHECK:         recurrence 1: 2 ops, 0 seq buffer(s) (0 slot(s)), 0 fifo buffer(s) (0 slot(s)), 0 init token(s)
// CHECK:         II: 0.00  |  if fifo were transparent: 0.00

// The same loop with the index buffer typed `fifo`. As lowered it still costs a
// cycle, so the II is unchanged; were fifo buffers lowered transparently, as
// the dialect documents, the recurrence bound would halve. The gap between the
// two figures is exactly the latency the lowering leaves on the table.

// CHECK-LABEL: handshake.func @fifoRecurrence: 2 recurrence(s)
// CHECK:         recurrence 0: 6 ops, 1 seq buffer(s) (1 slot(s)), 1 fifo buffer(s) (1 slot(s)), 1 init token(s)
// CHECK:         II: 1.00  |  if fifo were transparent: 0.50
// CHECK:         slots over whole component: 2 (as lowered), 1 (if fifo were transparent)

// JSON: "function": "fifoRecurrence"
// JSON: "ii": 1
// JSON: "ii-if-fifo-transparent": 0.5
handshake.func @fifoRecurrence(%arg0: none, ...) -> (i32, none) {
  %limit = constant %arg0 {value = 100 : i32} : i32
  %zero = constant %arg0 {value = 0 : i32} : i32
  %one = constant %arg0 {value = 1 : i32} : i32
  %prime = buffer [1] seq %cond {initValues = [0]} : i1
  %idx = mux %prime [%zero, %next] : i1, i32
  %ctrl = mux %prime [%arg0, %trueCtrl] : i1, none
  %idxBuf = buffer [1] fifo %idx : i32
  %cond = arith.cmpi slt, %idxBuf, %limit : i32
  %trueIdx, %falseIdx = cond_br %cond, %idxBuf : i32
  %trueCtrl, %falseCtrl = cond_br %cond, %ctrl : none
  %next = arith.addi %trueIdx, %one : i32
  return %falseIdx, %falseCtrl : i32, none
}

// A purely feed-forward function has no recurrences at all, however many
// buffers it carries.

// CHECK-LABEL: handshake.func @noRecurrence: 0 recurrence(s)
handshake.func @noRecurrence(%arg0: i32, %arg1: none, ...) -> (i32, none) {
  %0 = buffer [2] seq %arg0 : i32
  %1 = arith.addi %0, %0 : i32
  return %1, %arg1 : i32, none
}
