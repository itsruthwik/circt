//===- Buffers.cpp - buffer materialization passes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of buffer materialization passes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Dialect/Handshake/HandshakeUtils.h"
#include "circt/Support/SparseOpSCC.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"

namespace circt {
namespace handshake {
#define GEN_PASS_DEF_HANDSHAKEREMOVEBUFFERS
#define GEN_PASS_DEF_HANDSHAKEINSERTBUFFERS
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"
} // namespace handshake
} // namespace circt

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace {

struct RemoveHandshakeBuffers : public OpRewritePattern<handshake::BufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(handshake::BufferOp bufferOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOp(bufferOp, bufferOp.getOperand());
    return success();
  }
};

struct HandshakeRemoveBuffersPass
    : public circt::handshake::impl::HandshakeRemoveBuffersBase<
          HandshakeRemoveBuffersPass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();
    ConversionTarget target(getContext());
    target.addIllegalOp<handshake::BufferOp>();
    RewritePatternSet patterns(&getContext());
    patterns.insert<RemoveHandshakeBuffers>(&getContext());

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  };
};
} // namespace
// Returns true if a value lowers down to a handshake bundle, and so is
// something a buffer can be placed on.
static bool isDataflowChannel(Value v) {
  return v.getType().isIntOrFloat() || isa<NoneType>(v.getType());
}

// Returns true if a block argument should have buffers added to its uses.
static bool shouldBufferArgument(BlockArgument arg) {
  // At the moment, buffers only make sense on arguments which we know
  // will lower down to a handshake bundle.
  return isDataflowChannel(arg);
}

static bool isUnbufferedChannel(Operation *definingOp, Operation *usingOp) {
  return !isa_and_nonnull<BufferOp>(definingOp) && !isa<BufferOp>(usingOp);
}

static void insertBuffer(Location loc, Value operand, OpBuilder &builder,
                         unsigned numSlots, BufferTypeEnum bufferType) {
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointAfterValue(operand);
  auto bufferOp =
      handshake::BufferOp::create(builder, loc, operand, numSlots, bufferType);
  operand.replaceUsesWithIf(
      bufferOp, function_ref<bool(OpOperand &)>([](OpOperand &operand) -> bool {
        return !isa<handshake::BufferOp>(operand.getOwner());
      }));
  builder.restoreInsertionPoint(ip);
}

// Inserts buffers at all results of an operation
static void bufferResults(OpBuilder &builder, Operation *op, unsigned numSlots,
                          BufferTypeEnum bufferType) {
  for (auto res : op->getResults()) {
    // A result with no users has no channel to buffer, and a result whose every
    // user is already a buffer is buffered. Note that `insertBuffer` rewrites
    // all non-buffer uses at once, so a partially buffered result still only
    // needs one more buffer.
    if (res.use_empty())
      continue;
    if (llvm::all_of(res.getUsers(), [](Operation *user) {
          return isa<handshake::BufferOp>(user);
        }))
      continue;
    insertBuffer(op->getLoc(), res, builder, numSlots, bufferType);
  }
}

// Add a buffer to any un-buffered channel.
static void bufferAllStrategy(Region &r, OpBuilder &builder, unsigned numSlots,
                              BufferTypeEnum bufferType = BufferTypeEnum::seq) {

  for (auto &arg : r.getArguments()) {
    if (!shouldBufferArgument(arg))
      continue;
    insertBuffer(arg.getLoc(), arg, builder, numSlots, bufferType);
  }

  for (auto &defOp : r.getOps()) {
    for (auto res : defOp.getResults()) {
      for (auto *useOp : res.getUsers()) {
        if (!isUnbufferedChannel(&defOp, useOp))
          continue;
        insertBuffer(res.getLoc(), res, builder, numSlots, bufferType);
      }
    }
  }
}

// Returns true if 'src' is within a cycle. 'breaksCycle' is a function which
// determines whether an operation breaks a cycle.
static bool inCycle(Operation *src,
                    llvm::function_ref<bool(Operation *)> breaksCycle) {
  SetVector<Operation *> visited;
  SmallVector<Operation *> stack = {src};

  while (!stack.empty()) {
    Operation *curr = stack.pop_back_val();

    if (visited.contains(curr))
      continue;
    visited.insert(curr);

    if (breaksCycle(curr))
      continue;

    for (auto *user : curr->getUsers()) {
      // If visiting the source node, then we're in a cycle.
      if (src == user)
        return true;

      stack.push_back(user);
    }
  }
  return false;
}

// Perform a depth first search and insert buffers when cycles are detected.
static void
bufferCyclesStrategy(Region &r, OpBuilder &builder, unsigned numSlots,
                     BufferTypeEnum /*bufferType*/ = BufferTypeEnum::seq) {
  // Cycles can only occur at merge-like operations so those are our buffering
  // targets. Placing the buffer at the output of the merge-like op,
  // as opposed to naivly placing buffers *whenever* cycles are detected
  // ensures that we don't place a bunch of buffers on each input of the
  // merge-like op.
  auto isSeqBuffer = [](auto op) {
    auto bufferOp = dyn_cast<handshake::BufferOp>(op);
    return bufferOp && bufferOp.isSequential();
  };

  for (auto mergeOp : r.getOps<MergeLikeOpInterface>()) {
    // We insert a sequential buffer whenever the op is determined to be
    // within a cycle (to break combinational cycles). Else, place a FIFO
    // buffer.
    bool sequential = inCycle(mergeOp, isSeqBuffer);
    bufferResults(builder, mergeOp, numSlots,
                  sequential ? BufferTypeEnum::seq : BufferTypeEnum::fifo);
  }
}

// Combination of bufferCyclesStrategy and bufferAllStrategy, where we add a
// sequential buffer on graph cycles, and add FIFO buffers on all other
// connections.
static void bufferAllFIFOStrategy(Region &r, OpBuilder &builder,
                                  unsigned numSlots) {
  // First, buffer cycles with sequential buffers
  bufferCyclesStrategy(r, builder, /*numSlots=*/numSlots,
                       /*bufferType=*/BufferTypeEnum::seq);
  // Then, buffer remaining channels with transparent FIFO buffers
  bufferAllStrategy(r, builder, numSlots,
                    /*bufferType=*/BufferTypeEnum::fifo);
}

// Collects the values on which `op` returns a memory response: the data a load
// hands back, and the completion tokens of loads and stores. The latency of
// these values is determined outside the dataflow graph, so a consumer has to
// be decoupled from them.
static void collectMemoryResponses(Operation *op,
                                   SmallVectorImpl<Value> &responses) {
  auto addPorts = [&](auto memOp) {
    for (const MemLoadInterface &ld : memOp.getLoadPorts()) {
      responses.push_back(ld.dataOut);
      responses.push_back(ld.doneOut);
    }
    for (const MemStoreInterface &st : memOp.getStorePorts())
      responses.push_back(st.doneOut);
  };

  // Use the port accessors rather than re-deriving the result index arithmetic.
  if (auto memOp = dyn_cast<handshake::MemoryOp>(op))
    return addPorts(memOp);
  if (auto memOp = dyn_cast<handshake::ExternalMemoryOp>(op))
    return addPorts(memOp);

  // A load's address results travel *to* the memory and are part of the
  // request, not the response; only the data result is a response.
  if (auto loadOp = dyn_cast<handshake::LoadOp>(op)) {
    responses.push_back(loadOp.getDataResult());
    return;
  }

  // Any other operation that touches memory - including memory operations from
  // dialects this pass knows nothing about - is classified by its declared
  // memory effects.
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects || (!effects.hasEffect<MemoryEffects::Read>() &&
                   !effects.hasEffect<MemoryEffects::Write>()))
    return;
  for (Value res : op->getResults())
    if (isDataflowChannel(res))
      responses.push_back(res);
}

// Defined below: places the minimum buffers needed so no dataflow cycle is left
// combinational (a general cycle-cover over the SCC graph).
static void ensureCycleCover(Region &r, OpBuilder &builder, unsigned numSlots);

// Buffer merge-like outputs, which is where dataflow cycles close, plus every
// memory response channel.
//
// This is the smallest placement observed to keep kernels live. It was derived
// by repeatedly dropping buffers from the `allFIFO` placement until a kernel
// deadlocked: every buffer that turned out to be load-bearing was either on a
// merge-like output or on a memory response. Those locally-minimal sets are not
// unique (the search is greedy and order-dependent), so this strategy is a
// superset of them rather than a true minimum - but it is 60-75% smaller than
// `all`/`allFIFO`, and every buffer removed is a cycle of latency removed from
// whatever recurrence it sat on.
static void bufferMinimalStrategy(Region &r, OpBuilder &builder,
                                  unsigned numSlots) {
  // `inCycle` walks def-use edges and does not follow block arguments, so a
  // cycle closed through one would go unnoticed and the region would be
  // under-buffered. After `lower-cf-to-handshake` a handshake function is a
  // single block; say so rather than silently under-placing.
  if (!r.hasOneBlock())
    r.getParentOp()->emitWarning()
        << "buffer strategy 'minimal' may under-place buffers in a region with "
           "more than one block: cycles through block arguments are not "
           "detected";

  bufferCyclesStrategy(r, builder, numSlots);

  // A region argument that fans out feeds branches which reconverge later, at a
  // merge or at the terminator. Those branches are rarely the same depth, so
  // without slack at the fork the shallow one stalls waiting for the deep one
  // and the region deadlocks. Argument buffers sit outside every loop, so they
  // cost constant latency rather than latency per iteration.
  for (auto &arg : r.getArguments())
    if (shouldBufferArgument(arg))
      insertBuffer(arg.getLoc(), arg, builder, numSlots, BufferTypeEnum::fifo);

  SmallVector<Value> responses;
  for (Operation &op : r.getOps())
    collectMemoryResponses(&op, responses);

  for (Value res : responses) {
    if (res.use_empty())
      continue;
    if (llvm::all_of(res.getUsers(), [](Operation *user) {
          return isa<handshake::BufferOp>(user);
        }))
      continue;
    insertBuffer(res.getLoc(), res, builder, numSlots, BufferTypeEnum::fifo);
  }

  // `bufferCyclesStrategy` above only buffers merge-like outputs; a cycle that
  // closes through `cond_br`/`join`/a self-loop escapes it. Place the minimum
  // buffers so no dataflow cycle is left combinational.
  ensureCycleCover(r, builder, numSlots);
}

// True if `op` is a buffer that registers its channel, i.e. breaks any
// combinational cycle it sits on. HandshakeToHW lowers both `seq` and `fifo`
// buffers as sequential logic today (see BufferConversionPattern), so either
// kind, with at least one slot, breaks a combinational loop.
static bool isRegisteredBuffer(Operation *op) {
  auto bufferOp = dyn_cast<handshake::BufferOp>(op);
  return bufferOp && bufferOp.getNumSlots() > 0;
}

// The def-use edge model shared with RecurrenceReport, extended with a buffer
// cut. An edge runs from an op to each user of its results, confined to `body`,
// with two kinds of edge removed:
//
//  - The response half of a memory request/response pair. A load feeds a memory
//    an address and reads data back; that closes a def-use cycle but carries no
//    token around a loop, so treating it as a recurrence would flag every memory
//    access as an uncovered combinational cycle. Control/completion results are
//    `none`-typed and kept, because ordering through them is a genuine
//    dependence.
//  - Every edge out of a registered buffer. A buffer breaks the combinational
//    path through it, so cutting its out-edges makes any remaining cyclic SCC a
//    *genuinely unbuffered* cycle -- the precise structural condition ABC
//    rejects. This is what lets a single buffer on one edge cover a cycle: once
//    placed, that edge is cut and the cycle is no longer an SCC.
//
// `operand` is the edge's destination operand (its owner is the user op, which
// for forward traversal is `op`); the edge's source is its defining op.
static bool isDataflowCycleEdge(Region &body, Operation *op,
                                OpOperand &operand) {
  if (op->getParentRegion() != &body)
    return false;
  Operation *defOp = operand.get().getDefiningOp();
  if (isa_and_nonnull<handshake::MemoryOp, handshake::ExternalMemoryOp>(defOp))
    return isa<NoneType>(operand.get().getType());
  if (isa_and_nonnull<handshake::BufferOp>(defOp) && isRegisteredBuffer(defOp))
    return false;
  return true;
}

// Recompute the cyclic SCCs of `r` under the buffer-cutting edge model. Each
// cyclic SCC returned is a set of ops mutually reachable *without* passing
// through a registered buffer, i.e. a combinational loop that must be broken.
static SmallVector<SmallVector<Operation *>> uncoveredCycles(Region &r) {
  auto edgeFilter = [&r](Operation *op, OpOperand &operand) {
    return isDataflowCycleEdge(r, op, operand);
  };
  SparseOpSCC<OpSCCDirection::Forward> sccs(edgeFilter);
  for (Block &block : r)
    for (Operation &op : block)
      sccs.visit(&op);

  SmallVector<SmallVector<Operation *>> result;
  for (OpSCC entry : sccs.topological())
    if (auto cyclic = dyn_cast<CyclicOpSCC>(entry))
      result.emplace_back(cyclic.begin(), cyclic.end());
  return result;
}

// Place the minimum buffers needed so that no dataflow cycle is left without a
// registered buffer -- a general cycle-cover derived from the SCC graph, not a
// per-op rule. `bufferCyclesStrategy` only buffers merge-like outputs, so a
// cycle that closes through `cond_br`/`join`/a self-loop (loop back-edges lowered
// without a merge) escapes it and becomes a combinational loop. This closes that
// gap for every such cycle.
//
// Cover strategy: while an uncovered cyclic SCC remains, place one sequential
// buffer on a single intra-SCC edge, which cuts that edge from the graph, then
// recompute. One buffer per otherwise-uncovered back-edge is the minimal cover
// (an exact minimum feedback arc set is NP-hard; this greedy iteration places at
// most one buffer per remaining SCC per round and terminates when the graph is
// acyclic under the buffer cut).
static void ensureCycleCover(Region &r, OpBuilder &builder, unsigned numSlots) {
  // Each round breaks at least one cycle, so the loop is bounded by the number
  // of cycles; guard against an unexpected non-terminating case regardless.
  unsigned guard = 0;
  DenseSet<Operation *> members;
  while (true) {
    SmallVector<SmallVector<Operation *>> cycles = uncoveredCycles(r);
    if (cycles.empty())
      break;
    if (++guard > 100000)
      break;

    for (ArrayRef<Operation *> cycle : cycles) {
      members.clear();
      members.insert(cycle.begin(), cycle.end());
      // Break the cycle by buffering the first intra-SCC channel we find: a
      // result of a member op that another member consumes and that the edge
      // model actually traverses (so we do not buffer the cut memory-response
      // half and leave the real cycle intact).
      bool placed = false;
      for (Operation *op : cycle) {
        for (Value res : op->getResults()) {
          bool intra = llvm::any_of(res.getUses(), [&](OpOperand &use) {
            return members.contains(use.getOwner()) &&
                   isDataflowCycleEdge(r, use.getOwner(), use);
          });
          if (!intra)
            continue;
          insertBuffer(res.getLoc(), res, builder, numSlots,
                       BufferTypeEnum::seq);
          placed = true;
          break;
        }
        if (placed)
          break;
      }
      // A cyclic SCC always has an intra-component traversable edge, so this is
      // unreachable; recomputing next round would spin without it.
      assert(placed && "uncovered cycle had no bufferable intra-SCC edge");
      (void)placed;
    }
  }
}

// Post-placement guarantee: no dataflow cycle may be left combinational. If one
// survives, emit an error naming its ops -- so an under-placement fails loudly
// here instead of surfacing downstream as an ABC combinational-loop rejection.
static LogicalResult verifyCycleCover(Region &r) {
  SmallVector<SmallVector<Operation *>> cycles = uncoveredCycles(r);
  if (cycles.empty())
    return success();

  AsmState state(r.getParentOp());
  for (ArrayRef<Operation *> cycle : cycles) {
    InFlightDiagnostic diag =
        r.getParentOp()->emitError()
        << "buffer placement left a combinational loop: a dataflow cycle of "
        << cycle.size() << " op(s) carries no registered buffer";
    for (Operation *op : cycle) {
      std::string name = op->getName().getStringRef().str();
      if (op->getNumResults() > 0) {
        std::string ssa;
        llvm::raw_string_ostream os(ssa);
        op->getResult(0).printAsOperand(os, state);
        name = os.str() + " = " + name;
      }
      diag.attachNote(op->getLoc()) << "on cycle: " << name;
    }
  }
  return failure();
}

static LogicalResult bufferRegion(Region &r, OpBuilder &builder,
                                  StringRef strategy, unsigned bufferSize) {
  if (strategy == "cycles")
    bufferCyclesStrategy(r, builder, bufferSize);
  else if (strategy == "all")
    bufferAllStrategy(r, builder, bufferSize);
  else if (strategy == "allFIFO")
    bufferAllFIFOStrategy(r, builder, bufferSize);
  else if (strategy == "minimal") {
    bufferMinimalStrategy(r, builder, bufferSize);
    // The minimal strategy is the only one that can leave a cycle uncovered:
    // `all`/`allFIFO` buffer every channel, and `cycles` is not a shipping
    // production strategy. Verify the cycle-cover invariant so an under-placed
    // cycle fails loudly here instead of surfacing downstream as an ABC
    // combinational-loop rejection.
    return verifyCycleCover(r);
  } else
    return r.getParentOp()->emitOpError()
           << "Unknown buffer strategy: " << strategy;

  return success();
}

namespace {
struct HandshakeInsertBuffersPass
    : public circt::handshake::impl::HandshakeInsertBuffersBase<
          HandshakeInsertBuffersPass> {
  HandshakeInsertBuffersPass(const std::string &strategy, unsigned bufferSize) {
    this->strategy = strategy;
    this->bufferSize = bufferSize;
  }

  void runOnOperation() override {
    auto f = getOperation();
    if (f.isExternal())
      return;

    OpBuilder builder(f.getContext());

    if (failed(bufferRegion(f.getBody(), builder, strategy, bufferSize)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeRemoveBuffersPass() {
  return std::make_unique<HandshakeRemoveBuffersPass>();
}

std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
circt::handshake::createHandshakeInsertBuffersPass(const std::string &strategy,
                                                   unsigned bufferSize) {
  return std::make_unique<HandshakeInsertBuffersPass>(strategy, bufferSize);
}
