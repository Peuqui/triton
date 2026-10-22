#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H

#include "mlir/IR/Operation.h"

namespace mlir {
namespace triton {
namespace NVIDIA {

/// Return true if we can skip a barrier synchronization between two operations
/// even if they access the same shared memory.
bool canSkipBarSync(Operation *before, Operation *after);
// Barrier filter before sm_80 (register-staged shared buffers of dot operands).
bool canSkipRegisterStagedBarSync(Operation *before, Operation *after);
} // namespace NVIDIA
} // namespace triton
} // namespace mlir

#endif // TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H
