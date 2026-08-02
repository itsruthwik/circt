//===- RecurrenceReport.cpp - Handshake recurrence analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Reports the dataflow recurrences (cycles) of a handshake.func and the buffers
// placed on them.
//
// A handshake circuit's throughput is bounded by its recurrences: a loop-carried
// dependence can only produce a new token once the previous one has travelled
// all the way around the cycle. Every buffer on that path costs cycles, so the
// achievable initiation interval of a recurrence is determined by the buffers it
// carries, not by the buffers placed elsewhere in the circuit.
//
// This pass is purely observational -- it never mutates the IR. It exists to
// answer, with evidence rather than inspection:
//   - which recurrences a function actually has,
//   - how many buffers each carries and of which kind,
//   - the initiation interval those buffers impose today, and
//   - the initiation interval that would be achievable if `fifo` buffers were
//     lowered transparently, as the Handshake dialect documents them to be.
//
// Note that HandshakeToHW currently lowers `fifo` buffers as sequential buffers
// (see BufferConversionPattern, "For now, always build seq buffers"), so the
// two II figures differ only in what the hardware *could* achieve, not in what
// it does achieve today.
//
// The II is a maximum cycle ratio over the cycles of a component, not a sum of
// the buffers in it: a component generally contains many distinct cycles, and
// summing over all of them badly overstates any single one. What the pass does
// not do is claim deadlock -- see InitiationInterval for why that conclusion is
// not supportable from the local def-use graph.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/SparseOpSCC.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

namespace circt {
namespace handshake {
#define GEN_PASS_DEF_HANDSHAKERECURRENCEREPORT
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"
} // namespace handshake
} // namespace circt

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace {

/// The buffers carried by a single recurrence, and the latency they imply.
struct BufferInventory {
  unsigned seqBuffers = 0;
  unsigned fifoBuffers = 0;
  /// Slots contributed by sequential buffers. Each slot is one pipeline stage.
  unsigned seqSlots = 0;
  /// Slots contributed by transparent-typed (`fifo`) buffers.
  unsigned fifoSlots = 0;
  /// Number of initial tokens present on the cycle, i.e. buffer slots carrying
  /// an `initValues` entry.
  unsigned initTokens = 0;

  /// Latency around the cycle as lowered today: every buffer slot costs a
  /// cycle, because HandshakeToHW builds sequential logic for both buffer
  /// types.
  unsigned latencyAsLowered() const { return seqSlots + fifoSlots; }

  /// Latency around the cycle if `fifo` buffers were lowered transparently, as
  /// the dialect documents. Only sequential slots would then cost a cycle.
  unsigned latencyIfFifoTransparent() const { return seqSlots; }
};

/// The recurrence-constrained initiation interval, i.e. the maximum over the
/// component's cycles of (latency around the cycle) / (initial tokens on the
/// cycle).
///
/// This is the bound the recurrences impose, in the sense of the RecMII of
/// modulo scheduling; the achievable II is the maximum of this, any resource
/// bound, and one. A value below one therefore does not promise more than one
/// iteration per cycle -- it says the recurrence is not the binding constraint.
/// A value of zero means the cycle carries no buffer at all, which is a
/// combinational loop.
///
/// A strongly-connected component generally contains many distinct cycles, so
/// the buffers summed over the whole component are an upper bound on any single
/// cycle and must not be mistaken for an initiation interval. This is the real
/// figure: the ratio is maximised over the cycles of the component.
/// Note the deliberate limitation: no claim is made about deadlock. Tokens
/// enter a handshake cycle from the control network in ways that are not
/// determinable from the local def-use graph -- a memory whose first access is
/// issued from outside the loop, for instance, primes a cycle with no locally
/// visible token source. A cycle for which no token source can be identified is
/// therefore reported as such and its II left uncomputed, rather than being
/// declared deadlocked on evidence that does not support the conclusion.
struct InitiationInterval {
  /// True if some cycle has no structurally identifiable initial token, leaving
  /// its II undetermined. This is a limitation of the analysis, not a finding
  /// about the circuit.
  bool noTokenSource = false;
  /// The maximum cycle ratio. Only meaningful when `noTokenSource` is false.
  double value = 0.0;
};

/// One recurrence: a cyclic strongly-connected component of the dataflow graph.
struct Recurrence {
  SmallVector<Operation *> ops;
  BufferInventory buffers;
  /// II as the circuit is lowered today (every buffer slot costs a cycle).
  InitiationInterval ii;
  /// II if `fifo` buffers were lowered transparently, as the dialect documents.
  InitiationInterval iiIfFifoTransparent;
  /// Channels closing this recurrence: values whose producer and at least one
  /// consumer both lie in this component. A buffer placed on one of these adds
  /// to the recurrence's II; a buffer placed anywhere else cannot.
  SmallVector<Value> channels;
};

/// The recurrences of one function, plus the channels they span.
struct FuncRecurrences {
  SmallVector<Recurrence> recurrences;
  /// Every channel on any recurrence, deduplicated.
  SetVector<Value> onRecurrence;
  /// Channels on no recurrence. Buffering these trades area for slack without
  /// touching any initiation interval.
  SmallVector<Value> offRecurrence;
};

/// Print `v` as it appears in the IR (`%7`, `%arg0`, ...), so a reported channel
/// can be found in the function it came from.
static std::string ssaName(Value v, AsmState &state) {
  std::string name;
  llvm::raw_string_ostream os(name);
  v.printAsOperand(os, state);
  return os.str();
}

/// Collect the buffers carried by a set of ops forming a recurrence.
static BufferInventory inventoryBuffers(ArrayRef<Operation *> ops) {
  BufferInventory inv;
  for (Operation *op : ops) {
    auto bufferOp = dyn_cast<handshake::BufferOp>(op);
    if (!bufferOp)
      continue;
    unsigned slots = static_cast<unsigned>(bufferOp.getNumSlots());
    if (bufferOp.isSequential()) {
      ++inv.seqBuffers;
      inv.seqSlots += slots;
    } else {
      ++inv.fifoBuffers;
      inv.fifoSlots += slots;
    }
    if (auto initValues = bufferOp.getInitValues())
      inv.initTokens += initValues->size();
  }
  return inv;
}

/// Latency contributed by an operation, in cycles.
///
/// Only buffers are stateful; every other handshake operation is combinational
/// and contributes nothing. `fifoIsTransparent` selects whether a `fifo`-typed
/// buffer costs its slots (how HandshakeToHW lowers it today) or nothing (what
/// the dialect documents it to mean).
static unsigned opLatency(Operation *op, bool fifoIsTransparent) {
  auto bufferOp = dyn_cast<handshake::BufferOp>(op);
  if (!bufferOp)
    return 0;
  if (!bufferOp.isSequential() && fifoIsTransparent)
    return 0;
  return static_cast<unsigned>(bufferOp.getNumSlots());
}

/// Initial tokens injected at an operation, for a cycle contained in `sccOps`.
///
/// Two things put a token on a cycle at reset:
///   - a buffer carrying `initValues`, which comes out of reset already holding
///     one token per initialised slot, and
///   - a merge-like operation with a data input from outside the cycle, which
///     receives the loop-entry token from there. This is how a lowered loop is
///     primed: the priming register selects the external arm on the first
///     iteration.
static unsigned opInitTokens(Operation *op,
                             const DenseSet<Operation *> &sccOps) {
  if (auto bufferOp = dyn_cast<handshake::BufferOp>(op))
    if (auto initValues = bufferOp.getInitValues())
      return initValues->size();

  if (auto mergeLike = dyn_cast<handshake::MergeLikeOpInterface>(op)) {
    for (Value operand : mergeLike.getDataOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !sccOps.contains(defOp))
        return 1;
    }
  }
  return 0;
}

/// A recurrence's cycles, as an index-based graph over the SCC's operations.
struct CycleGraph {
  /// Per node: latency and initial tokens, both attributed to outgoing edges.
  SmallVector<unsigned> latency;
  SmallVector<unsigned> tokens;
  /// Directed edges (source, destination) between nodes of the component.
  SmallVector<std::pair<unsigned, unsigned>> edges;
};

/// Build the cycle graph of a recurrence.
static CycleGraph buildCycleGraph(ArrayRef<Operation *> ops,
                                  bool fifoIsTransparent) {
  DenseMap<Operation *, unsigned> index;
  DenseSet<Operation *> members;
  for (const auto &[i, op] : llvm::enumerate(ops)) {
    index[op] = i;
    members.insert(op);
  }

  CycleGraph graph;
  graph.latency.reserve(ops.size());
  graph.tokens.reserve(ops.size());
  for (Operation *op : ops) {
    graph.latency.push_back(opLatency(op, fifoIsTransparent));
    graph.tokens.push_back(opInitTokens(op, members));
  }

  for (const auto &[i, op] : llvm::enumerate(ops)) {
    // Mirror the edge model used to find the components: the response half of a
    // memory request/response pair is not a dataflow dependence.
    bool isMemory = isa<handshake::MemoryOp, handshake::ExternalMemoryOp>(op);
    for (Value result : op->getResults()) {
      if (isMemory && !isa<NoneType>(result.getType()))
        continue;
      for (Operation *user : result.getUsers()) {
        auto it = index.find(user);
        if (it != index.end())
          graph.edges.emplace_back(i, it->second);
      }
    }
  }
  return graph;
}

/// Return true if `graph` has a cycle whose total `latency - ratio * tokens` is
/// positive, i.e. a cycle whose latency-per-token exceeds `ratio`.
///
/// Bellman-Ford longest-path relaxation from an implicit source connected to
/// every node at distance zero; a node still improving after |V| rounds lies on
/// a positive-weight cycle.
static bool hasCycleAboveRatio(const CycleGraph &graph, double ratio) {
  SmallVector<double> dist(graph.latency.size(), 0.0);
  for (size_t round = 0, n = graph.latency.size(); round <= n; ++round) {
    bool improved = false;
    for (auto [src, dst] : graph.edges) {
      double weight = graph.latency[src] - ratio * graph.tokens[src];
      if (dist[src] + weight > dist[dst] + 1e-9) {
        dist[dst] = dist[src] + weight;
        improved = true;
      }
    }
    if (!improved)
      return false;
    // Still improving after a full pass over every node: positive cycle.
    if (round == n)
      return true;
  }
  return false;
}

/// Compute the maximum cycle ratio of a recurrence -- its initiation interval.
///
/// Lawler's parametric search: the answer is the unique ratio at which cycles
/// stop having positive `latency - ratio * tokens` weight. A cycle with no
/// initial token keeps positive weight for every ratio, which is how the
/// no-token-source case is detected.
static InitiationInterval computeII(ArrayRef<Operation *> ops,
                                    bool fifoIsTransparent) {
  CycleGraph graph = buildCycleGraph(ops, fifoIsTransparent);

  double totalLatency = 0.0;
  for (unsigned latency : graph.latency)
    totalLatency += latency;

  InitiationInterval result;
  // A zero-latency recurrence is combinational; it has no II to speak of, and
  // the search below would not terminate meaningfully.
  if (totalLatency == 0.0)
    return result;

  // Above the total latency of the component no cycle can still have positive
  // weight unless it carries no token at all, in which case its ratio is
  // unbounded and the II cannot be determined from structure alone.
  double upper = totalLatency + 1.0;
  if (hasCycleAboveRatio(graph, upper)) {
    result.noTokenSource = true;
    return result;
  }

  double lower = 0.0;
  // Enough iterations to pin the ratio far below any reporting precision.
  for (unsigned i = 0; i < 60; ++i) {
    double mid = (lower + upper) / 2.0;
    if (hasCycleAboveRatio(graph, mid))
      lower = mid;
    else
      upper = mid;
  }
  // The ratio is a rational of small integers, so snap away the binary-search
  // residue rather than reporting 3.9999999989999995 for 4.
  result.value = std::round(upper * 1e6) / 1e6;
  return result;
}

/// Find the dataflow recurrences of `func`.
///
/// Nodes are operations; an edge runs from an op to each user of its results.
/// The traversal is confined to the function body so that it neither escapes
/// into enclosing regions nor descends into unrelated ones.
static FuncRecurrences findRecurrences(handshake::FuncOp func) {
  Region &body = func.getBody();

  auto isDataflowEdge = [&body](Operation *op, OpOperand &operand) {
    // Confine the traversal to this function body. SparseOpSCC would otherwise
    // ascend from / descend into regions without regard for the parent op.
    if (op->getParentRegion() != &body)
      return false;

    // A memory op models a request and its response as a single operation: a
    // load feeds it an address and reads back data. In the def-use graph that
    // closes a cycle, but it is not a feedback loop -- no token travels around
    // it, and treating it as a recurrence makes every memory access look like a
    // deadlocked zero-token cycle. Cut the response half of the pair. Control
    // and completion results are `none`-typed and are kept, because ordering
    // through them is a genuine dependence.
    Operation *defOp = operand.get().getDefiningOp();
    if (isa_and_nonnull<handshake::MemoryOp, handshake::ExternalMemoryOp>(defOp))
      return isa<NoneType>(operand.get().getType());

    return true;
  };

  SparseOpSCC<OpSCCDirection::Forward> sccs(isDataflowEdge);
  for (Block &block : body)
    for (Operation &op : block)
      sccs.visit(&op);

  FuncRecurrences result;
  for (OpSCC entry : sccs.topological()) {
    auto cyclic = dyn_cast<CyclicOpSCC>(entry);
    if (!cyclic)
      continue;
    Recurrence rec;
    rec.ops.assign(cyclic.begin(), cyclic.end());
    rec.buffers = inventoryBuffers(rec.ops);
    rec.ii = computeII(rec.ops, /*fifoIsTransparent=*/false);
    rec.iiIfFifoTransparent = computeII(rec.ops, /*fifoIsTransparent=*/true);

    // A channel closes the recurrence when its producer and one of its
    // consumers are both members. Mirror the edge model used to find the
    // components, or the response half of a memory pair would be counted as
    // part of a cycle it does not actually close.
    DenseSet<Operation *> members(rec.ops.begin(), rec.ops.end());
    for (Operation *op : rec.ops) {
      bool isMemory = isa<handshake::MemoryOp, handshake::ExternalMemoryOp>(op);
      for (Value res : op->getResults()) {
        if (isMemory && !isa<NoneType>(res.getType()))
          continue;
        if (llvm::any_of(res.getUsers(), [&](Operation *user) {
              return members.contains(user);
            })) {
          rec.channels.push_back(res);
          result.onRecurrence.insert(res);
        }
      }
    }
    result.recurrences.push_back(std::move(rec));
  }

  // Everything else that can carry a token is off every recurrence.
  auto isChannel = [](Value v) {
    return v.getType().isIntOrFloat() || isa<NoneType>(v.getType());
  };
  for (BlockArgument arg : body.getArguments())
    if (isChannel(arg) && !result.onRecurrence.contains(arg))
      result.offRecurrence.push_back(arg);
  for (Block &block : body)
    for (Operation &op : block)
      for (Value res : op.getResults())
        if (isChannel(res) && !res.use_empty() &&
            !result.onRecurrence.contains(res))
          result.offRecurrence.push_back(res);

  return result;
}

/// Render one recurrence as JSON.
static llvm::json::Value toJSON(const Recurrence &rec, unsigned id,
                                AsmState &state) {
  llvm::json::Array opNames;
  for (Operation *op : rec.ops) {
    // Name the op by its first result where it has one, so a reported
    // recurrence can be traced back to specific values in the function.
    std::string name = op->getName().getStringRef().str();
    if (op->getNumResults() > 0)
      name = ssaName(op->getResult(0), state) + " = " + name;
    opNames.push_back(std::move(name));
  }

  llvm::json::Array channelNames;
  for (Value v : rec.channels)
    channelNames.push_back(ssaName(v, state));

  // A null II means no token source could be identified, so the ratio is
  // undetermined -- not that the recurrence is known to deadlock.
  auto iiToJSON = [](const InitiationInterval &ii) -> llvm::json::Value {
    if (ii.noTokenSource)
      return nullptr;
    return ii.value;
  };

  const BufferInventory &inv = rec.buffers;
  return llvm::json::Object{
      {"id", id},
      {"size", static_cast<int64_t>(rec.ops.size())},
      {"ops", std::move(opNames)},
      {"channels", std::move(channelNames)},
      {"seq-buffers", inv.seqBuffers},
      {"fifo-buffers", inv.fifoBuffers},
      {"seq-slots", inv.seqSlots},
      {"fifo-slots", inv.fifoSlots},
      {"init-tokens", inv.initTokens},
      {"ii", iiToJSON(rec.ii)},
      {"ii-if-fifo-transparent", iiToJSON(rec.iiIfFifoTransparent)},
      {"scc-slots", inv.latencyAsLowered()},
      {"scc-slots-if-fifo-transparent", inv.latencyIfFifoTransparent()},
  };
}

/// Render one recurrence as human-readable text.
static void printRecurrence(llvm::raw_ostream &os, const Recurrence &rec,
                            unsigned id, AsmState &state) {
  auto printII = [&os](const InitiationInterval &ii) {
    if (ii.noTokenSource)
      os << "undetermined (no token source identified on some cycle)";
    else
      os << llvm::format("%.2f", ii.value);
  };

  const BufferInventory &inv = rec.buffers;
  os << "  recurrence " << id << ": " << rec.ops.size() << " ops, "
     << inv.seqBuffers << " seq buffer(s) (" << inv.seqSlots << " slot(s)), "
     << inv.fifoBuffers << " fifo buffer(s) (" << inv.fifoSlots
     << " slot(s)), " << inv.initTokens << " init token(s)\n";
  os << "    II: ";
  printII(rec.ii);
  os << "  |  if fifo were transparent: ";
  printII(rec.iiIfFifoTransparent);
  os << "\n";
  // Component-wide slot totals bound any single cycle from above; they are not
  // an initiation interval.
  os << "    slots over whole component: " << inv.latencyAsLowered()
     << " (as lowered), " << inv.latencyIfFifoTransparent()
     << " (if fifo were transparent)\n";
  os << "    channels closing it (" << rec.channels.size() << "):";
  for (Value v : rec.channels)
    os << " " << ssaName(v, state);
  os << "\n";
  os << "    ops:";
  for (Operation *op : rec.ops) {
    os << " ";
    if (op->getNumResults() > 0)
      os << ssaName(op->getResult(0), state) << "=";
    os << op->getName();
  }
  os << "\n";
}

struct HandshakeRecurrenceReportPass
    : public circt::handshake::impl::HandshakeRecurrenceReportBase<
          HandshakeRecurrenceReportPass> {
  using HandshakeRecurrenceReportBase<
      HandshakeRecurrenceReportPass>::HandshakeRecurrenceReportBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::raw_ostream &os = llvm::outs();
    bool emitJSON = format == "json";
    if (!emitJSON && format != "text") {
      module.emitError() << "unknown recurrence report format: " << format
                         << " (expected 'text' or 'json')";
      return signalPassFailure();
    }

    llvm::json::Array funcsJSON;
    for (auto func : module.getOps<handshake::FuncOp>()) {
      if (func.isExternal())
        continue;

      // SparseOpSCC does not traverse block arguments, so a recurrence that
      // closes through one would be missed. After CFToHandshake a function body
      // is a single block, but say so rather than under-reporting in silence.
      unsigned numBlocks =
          std::distance(func.getBody().begin(), func.getBody().end());
      if (numBlocks > 1)
        func.emitWarning()
            << "handshake.func has " << numBlocks
            << " blocks; recurrences closing through block arguments are not "
               "detected and this report may be incomplete";

      FuncRecurrences found = findRecurrences(func);
      ArrayRef<Recurrence> recurrences = found.recurrences;
      AsmState state(func);

      if (emitJSON) {
        llvm::json::Array recsJSON;
        for (const auto &[id, rec] : llvm::enumerate(recurrences))
          recsJSON.push_back(toJSON(rec, id, state));
        llvm::json::Array offJSON;
        for (Value v : found.offRecurrence)
          offJSON.push_back(ssaName(v, state));
        funcsJSON.push_back(llvm::json::Object{
            {"function", func.getName()},
            {"blocks", numBlocks},
            {"recurrences", std::move(recsJSON)},
            {"off-recurrence-channels", std::move(offJSON)},
        });
        continue;
      }

      os << "handshake.func @" << func.getName() << ": " << recurrences.size()
         << " recurrence(s)\n";
      for (const auto &[id, rec] : llvm::enumerate(recurrences))
        printRecurrence(os, rec, id, state);
      os << "  channels off every recurrence: " << found.offRecurrence.size()
         << " (buffering these cannot change any II)\n";
    }

    if (emitJSON)
      os << llvm::formatv("{0:2}", llvm::json::Value(std::move(funcsJSON)))
         << "\n";
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::handshake::createHandshakeRecurrenceReportPass() {
  return std::make_unique<HandshakeRecurrenceReportPass>();
}
