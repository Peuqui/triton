#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  OpBuilder &builder;
  Location loc;

public:
  GenericFMAVectorMultiplier(OpBuilder &builder, Location loc)
      : builder(builder), loc(loc) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    Value accum = c;
    Type accTy = accum.getType();
    // A half-precision accumulator sums its K products in fp32 and rounds
    // once, as tensor cores do; rounding after every product drifted by
    // several fp16 ulps at K = 64.
    bool widen = isa<Float16Type, BFloat16Type>(accTy);
    Type tgtTy = widen ? builder.getF32Type() : accTy;
    if (widen)
      accum = LLVM::FPExtOp::create(builder, loc, tgtTy, accum);
    for (auto it = llvm::zip(a, b).begin(); it != llvm::zip(a, b).end(); ++it) {
      Value aElem = std::get<0>(*it);
      Value bElem = std::get<1>(*it);
      assert(aElem.getType() == accTy);
      assert(bElem.getType() == accTy);
      if (widen) {
        aElem = LLVM::FPExtOp::create(builder, loc, tgtTy, aElem);
        bElem = LLVM::FPExtOp::create(builder, loc, tgtTy, bElem);
      }

      // to avoid: 'llvm.intr.fmuladd' op operand #0 must be floating point LLVM
      // type or LLVM dialect-compatible vector of floating point LLVM type, but
      // got 'i32'
      llvm::TypeSwitch<Type>(tgtTy)
          .Case<FloatType>([&](auto) {
            accum = LLVM::FMulAddOp::create(builder, loc, aElem, bElem, accum);
          })
          .Case<IntegerType>([&](auto) {
            accum = LLVM::AddOp::create(
                builder, loc, LLVM::MulOp::create(builder, loc, aElem, bElem),
                accum);
          });
    }
    if (widen)
      accum = LLVM::FPTruncOp::create(builder, loc, accTy, accum);
    return accum;
  }
};

} // namespace

LogicalResult convertFMADot(DotOp op, DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
