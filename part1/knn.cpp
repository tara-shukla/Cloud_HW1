#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}


// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T,int>>& items, int depth) {
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */

    if (items.empty()) return nullptr;

    sort(items.begin(), items.end(),
		[](auto& a, auto& b){
			return (a.first > b.first);
		});
    

    int n = items.size();

    //just doing this should still make the median rule go to the smaller one right?
    int medianIndex = n/2;
    
    std::vector<std::pair<Embedding_T,int>> leftTree(items.begin(), items.begin()+medianIndex);
    std::vector<std::pair<Embedding_T,int>> rightTree(items.begin()+medianIndex, items.end());
    
    Node* root = new Node{items[medianIndex].first, items[medianIndex].second};
    
    //build tree
    root->left = buildKD(leftTree, depth + 1);
    root->right = buildKD(rightTree, depth + 1);
    
    return root;
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */
    return;
}