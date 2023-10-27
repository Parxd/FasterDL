#ifndef FAST_STORAGE_HPP
#define FAST_STORAGE_HPP

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
    int inc_refcnt();
    int dec_refcnt();
private:
    size_t size;
    size_t refs;
};

}

#endif // STORAGE_HPP