#include "core/graph.h"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    // 更新前驱和后继关系
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        // 确保图是拓扑排序的，以便按计算顺序遍历
        if (!sorted) {
            topo_sort();
        }

        // 辅助函数：断开算子与图中其他元素的连接
        auto disconnectOperator = [&](const Operator &op) {
            for (auto pred : op->getPredecessors()) {
                pred->removeSuccessors(op);
            }
            for (auto succ : op->getSuccessors()) {
                succ->removePredecessors(op);
            }
            for (auto input : op->getInputs()) {
                input->removeTarget(op);
            }
            for (auto output : op->getOutputs()) {
                output->setSource(nullptr);
            }
        };

        // 辅助函数：检查 transpose 是否只交换最后两个维度
        auto isLastTwoDimsSwap = [](const std::vector<int> &perm) -> bool {
            for (size_t i = 0; i < perm.size(); ++i) {
                if (i == perm.size() - 2) {
                    if (perm[i] != (int)perm.size() - 1) return false;
                } else if (i == perm.size() - 1) {
                    if (perm[i] != (int)perm.size() - 2) return false;
                } else {
                    if (perm[i] != (int)i) return false;
                }
            }
            return true;
        };

        // 使用集合避免重复标记
        std::unordered_set<Operator> operatorsToRemoveSet;

        // 规则 1: 消除冗余的 transpose 对
        for (size_t i = 0; i < ops.size(); ++i) {
            auto op1 = ops[i];
            if (op1->getOpType() != OpType::Transpose) continue;
            if (operatorsToRemoveSet.count(op1)) continue; // 已标记移除

            auto transpose1 = as<TransposeObj>(op1);
            auto perm1 = transpose1->getPermute();
            auto output1 = op1->getOutput();
            if (!output1) continue;

            // 查找直接后继中的 transpose 算子
            auto successors = op1->getSuccessors();
            for (auto op2 : successors) {
                if (op2->getOpType() != OpType::Transpose) continue;
                if (operatorsToRemoveSet.count(op2)) continue; // 已标记移除

                auto transpose2 = as<TransposeObj>(op2);
                auto perm2 = transpose2->getPermute();
                if (perm1 != perm2) continue; // 需要相同的 permutation（互逆）

                // 找到匹配的 transpose 对，将其绕过
                auto input1 = op1->getInputs(0);
                auto output2 = op2->getOutput();

                // 将 op2 的所有消费者重定向到 op1 的输入
                auto consumers = output2->getTargets();
                for (auto consumer : consumers) {
                    consumer->replaceInput(output2, input1);
                    output2->removeTarget(consumer);
                    input1->addTarget(consumer);
                }

                // 清理中间 tensor 的连接
                output1->removeTarget(op2);
                output1->setSource(nullptr);
                output2->setSource(nullptr);

                operatorsToRemoveSet.insert(op1);
                operatorsToRemoveSet.insert(op2);
                break; // 处理第一个匹配的后继
            }
        }

        // 规则 2: 将 transpose 合并到 matmul
        auto opsCopy = ops; // 复制列表，因为遍历过程中可能修改 ops
        for (auto op : opsCopy) {
            if (op->getOpType() != OpType::MatMul) continue;

            auto matmul = as<MatmulObj>(op);

            // 检查两个输入
            for (int idx = 0; idx < 2; ++idx) {
                auto input = op->getInputs(idx);
                auto source = input->getSource();
                if (!source || source->getOpType() != OpType::Transpose) continue;
                if (operatorsToRemoveSet.count(source)) continue; // 已标记移除

                auto transpose = as<TransposeObj>(source);
                auto perm = transpose->getPermute();
                if (!isLastTwoDimsSwap(perm)) continue;

                // 合并 transpose 到 matmul
                auto transposeInput = source->getInputs(0);

                // 更新 matmul 的输入
                op->replaceInput(input, transposeInput);
                input->removeTarget(op);
                transposeInput->addTarget(op);

                // 设置 transA/transB 属性
                if (idx == 0) {
                    matmul->setTransA(!matmul->getTransA());
                } else {
                    matmul->setTransB(!matmul->getTransB());
                }

                // 标记 transpose 等待移除
                operatorsToRemoveSet.insert(source);

                // 断开 transpose 输出的连接
                auto transposeOutput = source->getOutput();
                if (transposeOutput) {
                    transposeOutput->setSource(nullptr);
                }
            }
        }

        // 移除所有标记的算子
        for (auto op : operatorsToRemoveSet) {
            disconnectOperator(op);
            removeOperator(op);
        }

        // 清理孤立的 tensor（没有 source 且没有 target）
        for (auto it = tensors.begin(); it != tensors.end(); ) {
            auto tensor = *it;
            if (!tensor->getSource() && tensor->getTargets().empty()) {
                it = tensors.erase(it);
            } else {
                ++it;
            }
        }

        // 优化后图结构可能已改变，标记为未排序状态
        sorted = false;
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        vector<size_t> offsets(tensors.size());
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            offsets[i] = allocator.alloc(tensors[i]->getBytes());
        }
        void *base_ptr = allocator.getPtr();
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            Blob blob = make_ref<BlobObj>(runtime, (char *)base_ptr + offsets[i]);
            tensors[i]->setDataBlob(blob);
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini