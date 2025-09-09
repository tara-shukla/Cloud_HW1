#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>


template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
};

// dynamic vector: runtime-D (global, set once at startup)
inline size_t& runtime_dim() {
    static size_t d = 0;
    return d;
}

// variable-size vector: N-D
template <>
struct Embedding_T<std::vector<float>>
{
    static size_t Dim() { return runtime_dim(); }
    
    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        float s = 0;
        for (size_t i = 0; i < Dim(); ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};


// extract the “axis”-th coordinate or the scalar itself
template<typename T>
constexpr float getCoordinate(T const &e, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return e;          // scalar case
    } else {
        return e[axis];    // vector case
    }
}


// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;


/**
 * Builds a KD-tree from a vector of items,
 * where each item consists of an embedding and its associated index.
 * The splitting dimension is chosen based on the current depth.
 *
 * @param items A reference to a vector of pairs, each containing an embedding (Embedding_T)
 *              and an integer index.
 * @param depth The current depth in the tree, used to determine the splitting dimension (default is 0).
 * @return A pointer to the root node of the constructed KD-tree.
 */
// Build a balanced KD‐tree by splitting on median at each level.



template <typename T>
Node<T>* buildKD(std::vector<std::pair<T,int>>& items, int depth = 0)
{
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    */

    if (items.empty()) return nullptr;
    int axis = depth % static_cast<int>(Embedding_T<T>::Dim());

    // diff than part 1, we use depth 
    sort(items.begin(), items.end(),
		[&axis](auto& a, auto& b){
			return (getCoordinate(a.first, axis) < getCoordinate(b.first, axis));
		});
    

    int n = items.size();

    int medianIndex = (n-1)/2;
    
    auto leftTree = std::vector(items.begin(), items.begin()+medianIndex);
    auto rightTree = std::vector(items.begin()+medianIndex+1, items.end());
    
    auto* root = new Node{items[medianIndex].first, items[medianIndex].second};
    
    //build tree
    root->left = buildKD(leftTree, depth + 1);
    root->right = buildKD(rightTree, depth + 1);
    
    return root;
}

template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;


/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<
    PQItem,
    std::vector<PQItem>,
    std::less<PQItem>>;

/**
 * @brief Performs a k-nearest neighbors (k-NN) search on a KD-tree.
 *
 * This function recursively traverses the KD-tree starting from the given node,
 * searching for the K nearest neighbors to a target point. The results are maintained
 * in a max-heap, and an optional epsilon parameter can be used to allow for approximate
 * nearest neighbor search.
 *
 * @param node Pointer to the current node in the KD-tree.
 * @param depth Current depth in the KD-tree (used to determine splitting axis).
 * @param K Number of nearest neighbors to search for.
 * @param epsilon Approximation factor for the search (0 for exact search).
 * @param heap Reference to a max-heap that stores the current K nearest neighbors found.
 */
template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */


   if (node==nullptr){
        return;
   }

    // size_t my_size = Embedding_T<T>::Dim();
    int axis = depth %  static_cast<int> (Embedding_T<T>::Dim());

    //Compare the query point (Node<T>::queryEmbedding) to the current node’s point along the splitting axis.
    if (getCoordinate(Node<T>::queryEmbedding, axis) < getCoordinate(node->embedding, axis)){
        knnSearch(node->left, depth+1,K, heap);
    }
        
    else{
        knnSearch(node->right, depth+1,K, heap);
    }

    //now heap is updated w closer tree candidates
    //we check current node -- shd it be added to heap?

    if (heap.size()< static_cast<size_t>(K)){
        // heap.push({node->embedding::distance(Node<T>::queryEmbedding, node->embedding), node->idx});
        heap.push({Embedding_T<T>::distance(Node<T>::queryEmbedding, node->embedding), node->idx});
    }
    else if (Embedding_T<T>::distance(Node<T>::queryEmbedding, node->embedding) < heap.top().first){
        heap.pop();
        heap.push({Embedding_T<T>::distance(Node<T>::queryEmbedding, node->embedding), node->idx});
    }

    //if current node is not the worst node on the heap then we can't prune 
    //because something could still beat the worst node -- explore other
    //or if we still need to add candidates

    float planeDist = std::abs(getCoordinate(Node<T>::queryEmbedding, axis)-getCoordinate(node->embedding, axis));

    // if (heap.size()< static_cast<size_t>(K) || distance(heap.top().first, Node::queryEmbedding)> planeDist){

    if (heap.size()< static_cast<size_t>(K) || heap.top().first > planeDist){

        if (getCoordinate(Node<T>::queryEmbedding, axis) < getCoordinate(node->embedding, axis)){
            knnSearch(node->right, depth+1, K,heap);
        }
        else{
            knnSearch(node->left, depth+1, K, heap);
        }
    }
    return;
}