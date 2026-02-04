#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        used += size;

        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it)
        {
            size_t block_offset = it->first;
            size_t block_size = it->second;

            if (block_size >= size)
            {
                size_t allocated_offset = block_offset;

                // Update or remove the free block
                if (block_size > size)
                {
                    size_t new_block_offset = block_offset + size;
                    size_t new_block_size = block_size - size;
                    free_blocks.erase(it);
                    free_blocks[new_block_offset] = new_block_size;
                }
                else
                {
                    free_blocks.erase(it);
                }

                return allocated_offset;
            }
        }

        peak += size;

        return peak - size;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        used -= size;
        free_blocks[addr] = size;
        // 合并内存
        auto it = free_blocks.find(addr);
        // 向前合并
        if (it != free_blocks.begin())
        {
            auto prev = std::prev(it);
            if (prev->first + prev->second == addr)
            {
                prev->second += it->second;
                free_blocks.erase(it);
                it = prev;
            }
        }
        // 向后合并
        auto next = std::next(it);
        if (next != free_blocks.end() && it->first + it->second == next->first)
        {
            it->second += next->second;
            free_blocks.erase(next);
        }
        // 如果是最后一个 free block，尝试收缩 peak
        if (it->first + it->second == peak)
        {
            peak = it->first;
            free_blocks.erase(it);
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
