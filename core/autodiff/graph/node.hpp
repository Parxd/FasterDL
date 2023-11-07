class Node {
public:
    virtual ~Node() = default;
    virtual void forward() = 0;
    virtual void backward() = 0;
};
