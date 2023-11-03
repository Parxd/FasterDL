#ifndef FAST_STORAGE_HPP
#define FAST_STORAGE_HPP

#include "types.cuh"

namespace fast {
    template <typename dtype>
    class Storage {
    public:
        Storage();
        explicit Storage(Storage&);
        ~Storage();
        inline void destroy() {
            ~Storage();
        };
        void swap(dtype*);
        size_t inc_refcnt();
        size_t dec_refcnt();
    private:
        d_type buffer_ptr;
        size_t size;
        size_t refs;
    };
}

#endif // STORAGE_HPP