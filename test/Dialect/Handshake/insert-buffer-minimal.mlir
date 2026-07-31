// RUN: circt-opt -handshake-insert-buffers="strategy=minimal buffer-size=1" %s | circt-opt -handshake-insert-buffers="strategy=minimal buffer-size=1" | FileCheck %s

// The `minimal` strategy places a buffer on every merge-like output (where
// dataflow cycles close), on every region argument that can carry a token (a
// fan-out there feeds branches that reconverge at different depths), and on
// every memory response channel (load data and completion tokens) -- and
// nowhere else. Running the pass twice checks that it is idempotent.

// CHECK-LABEL: handshake.func @memoryResponses(
// CHECK:     buffer [1] fifo %arg1 : none
handshake.func @memoryResponses(%mem: memref<4xi32>, %ctrl: none, ...) -> none {
  // Both the load data and the completion token of the memory are responses.
  // CHECK:     %[[MEM:.*]]:2 = extmemory
  // CHECK-DAG: buffer [1] fifo %[[MEM]]#0 : i32
  // CHECK-DAG: buffer [1] fifo %[[MEM]]#1 : none
  %0:2 = extmemory[ld = 1, st = 0] (%mem : memref<4xi32>) (%addressResults) {id = 0 : i32} : (index) -> (i32, none)

  %idx = constant %ctrl {value = 0 : index} : index

  // The load's data result is a response and is buffered; its address results
  // travel *to* the memory and are part of the request, so they are not.
  // CHECK:     %[[DATA:.*]], %[[ADDR:.*]] = load
  // CHECK-NEXT: buffer [1] fifo %[[DATA]] : i32
  // Plain arithmetic is neither merge-like nor a memory response, so it is left
  // unbuffered -- and the load's address result, which travels *to* the memory
  // as part of the request, is never buffered either.
  // CHECK-NEXT: arith.muli
  // CHECK-NOT:  buffer
  %dataResult, %addressResults = load [%idx] %0#0, %ctrl : index, i32
  %sq = arith.muli %dataResult, %dataResult : i32
  sink %sq : i32

  return %0#1 : none
}

// CHECK-LABEL: handshake.func @cyclicMerge(
// CHECK:     buffer [1] fifo %arg0 : none
handshake.func @cyclicMerge(%ctrl: none, ...) -> i32 {
  %c0 = constant %ctrl {value = 0 : i32} : i32
  %c1 = constant %ctrl {value = 1 : i32} : i32
  %c8 = constant %ctrl {value = 8 : i32} : i32

  // The mux closes the accumulator cycle, so it gets a sequential buffer.
  // CHECK: %[[ACC:.*]] = mux
  // CHECK: buffer [1] seq %[[ACC]]
  %cond = buffer [1] seq %cmp {initValues = [0]} : i1
  %acc = mux %cond [%c0, %next] : i1, i32
  %next = arith.addi %acc, %c1 : i32
  %cmp = arith.cmpi slt, %acc, %c8 : i32
  %trueResult, %falseResult = cond_br %cmp, %acc : i32
  sink %trueResult : i32
  return %falseResult : i32
}
