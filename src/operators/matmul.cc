#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Tensor A = inputs[0];
        Tensor B = inputs[1];

        m = transA ? A->getDims()[A->getRank() - 1] : A->getDims()[A->getRank() - 2];
        k = transA ? A->getDims()[A->getRank() - 2] : A->getDims()[A->getRank() - 1];
        n = transB ? B->getDims()[B->getRank() - 2] : B->getDims()[B->getRank() - 1];

        Shape aShapeExtended = A->getDims();
        Shape bShapeExtended = B->getDims();
        aShapeExtended[A->getRank() - 2] = m;
        aShapeExtended[A->getRank() - 1] = n;
        bShapeExtended[B->getRank() - 2] = m;
        bShapeExtended[B->getRank() - 1] = n;

        Shape result = infer_broadcast(aShapeExtended, bShapeExtended);
        return vector<Shape>{result};
    }

} // namespace infini