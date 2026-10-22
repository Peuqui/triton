#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"

#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

using Coord = std::pair<int32_t, int32_t>;

// (row, col) that register `reg` of lane 0, warp 0 holds in `layout`. Lanes
// and warps only add offsets inside an MMA tile or move whole tiles, so the
// coordinates of lane 0 identify the operand slot of every register.
Coord registerCoord(const LinearLayout &layout, int reg, MLIRContext *ctx) {
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  auto out = layout.apply(
      {{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
  return {out[0].second, out[1].second};
}

// Operand values by their (row, col) coordinate for lane 0.
std::map<Coord, Value> operandByCoord(Value llOperand, RankedTensorType type,
                                      Location loc,
                                      ConversionPatternRewriter &rewriter) {
  auto elems = unpackLLElements(loc, llOperand, rewriter);
  auto layout = triton::gpu::toLinearLayout(type);
  std::map<Coord, Value> values;
  for (int reg = 0; reg < (int)elems.size(); ++reg)
    values[registerCoord(layout, reg, type.getContext())] = elems[reg];
  return values;
}

} // namespace

// D = A * B + C on Volta tensor cores with mma.sync.m8n8k4 (fp16 operands,
// fp32 accumulators). Register slots come from the MMAv1 linear layouts
// (see voltaMmaTile in LinearLayoutConversions.cpp): per 16x16 warp tile each
// quadpair runs one m8n8k4 per k step of 4.
LogicalResult convertMMA884(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto *ctx = op.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto aType = op.getA().getType();
  auto bType = op.getB().getType();
  auto dType = op.getD().getType();
  assert(aType.getElementType().isF16() && bType.getElementType().isF16() &&
         dType.getElementType().isF32() &&
         "Volta MMA lowers fp16 x fp16 -> fp32 only");

  auto aValues = operandByCoord(adaptor.getA(), aType, loc, rewriter);
  auto bValues = operandByCoord(adaptor.getB(), bType, loc, rewriter);

  // Accumulator registers per 16x16 tile, ordered as the eight fp32 results
  // of m8n8k4. The first three register bits of the MMAv1 layout are that
  // fragment index ((col & 1) | (row & 2) | (col & 4) for lane 0); the higher
  // bits count tiles. An output smaller than a tile folds its coordinates
  // (a 2x4 result maps row 2 onto row 0), so the slot comes from the register
  // index and each tile's origin from its first register.
  SmallVector<Value> acc = unpackLLElements(loc, adaptor.getC(), rewriter);
  auto dLayout = triton::gpu::toLinearLayout(dType);
  std::map<int, std::pair<Coord, std::array<int, 8>>> tiles;
  for (int reg = 0; reg < (int)acc.size(); ++reg) {
    auto &tile = tiles[reg >> 3];
    if ((reg & 7) == 0)
      tile.first = registerCoord(dLayout, reg, ctx);
    tile.second[reg & 7] = reg;
  }

  Type packTy = vec_ty(f16_ty, 2);
  auto pack = [&](Value lo, Value hi) {
    Value vec = b.undef(packTy);
    vec = b.insert_element(packTy, vec, lo, b.i32_val(0));
    vec = b.insert_element(packTy, vec, hi, b.i32_val(1));
    return b.bitcast(vec, i32_ty);
  };

  int K = aType.getShape()[1];
  Type retTy = LLVM::LLVMStructType::getLiteral(
      ctx, SmallVector<Type>(8, f32_ty));
  for (int k = 0; k < K; k += 4) {
    for (auto &[index, tile] : tiles) {
      auto &[origin, slots] = tile;
      auto [row, col] = origin;
      PTXBuilder builder;
      auto *dArgs = builder.newListOperand(8, "=f");
      auto *aArgs = builder.newListOperand(
          {{pack(aValues.at({row, k}), aValues.at({row, k + 1})), "r"},
           {pack(aValues.at({row, k + 2}), aValues.at({row, k + 3})), "r"}});
      auto *bArgs = builder.newListOperand(
          {{pack(bValues.at({k, col}), bValues.at({k + 1, col})), "r"},
           {pack(bValues.at({k + 2, col}), bValues.at({k + 3, col})), "r"}});
      auto *cArgs = builder.newListOperand();
      for (int i = 0; i < 8; ++i)
        cArgs->listAppend(builder.newOperand(acc[slots[i]], std::to_string(i)));
      auto &mma =
          *builder.create("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32");
      mma(dArgs, aArgs, bArgs, cArgs);
      Value res = builder.launch(rewriter, loc, retTy);
      for (int i = 0; i < 8; ++i)
        acc[slots[i]] = b.extract_val(f32_ty, res, i);
    }
  }

  Value result = packLLElements(loc, typeConverter, acc, rewriter, dType);
  rewriter.replaceOp(op, result);
  return success();
}
