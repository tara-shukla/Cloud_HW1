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
			return (a.first < b.first);
		});
    

    int n = items.size();

    int medianIndex = (n-1)/2;
    
    std::vector<std::pair<Embedding_T,int>> leftTree(items.begin(), items.begin()+medianIndex);
    std::vector<std::pair<Embedding_T,int>> rightTree(items.begin()+medianIndex+1, items.end());
    
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
   if (node==nullptr){
        return;
   }
    int axis = depth % Embedding_T<T>::Dim;

    //Compare the query point (Node<T>::queryEmbedding) to the current node’s point along the splitting axis.
    if getCoordinate(node->queryEmbedding, axis) < getCoordinate(node->embedding, axis){
        knnSearch(node->left, depth+1, heap);
    }
        
    else{
        knnSearch(node->right, depth+1, heap);
    }

    //now heap is updated w closer tree candidates
    //we check current node -- shd it be added to heap?

    if (heap.size()<K){
        heap.push(node->embedding);

    }
    else if (distance(node->queryEmbedding, node->embedding) < heap.top()){
        heap.pop();
        heap.push(node->embedding);
    }

    //if current node is not the worst node on the heap then we can't prune 
    //because something could still beat the worst node -- explore other
    //or if we still need to add candidates

    planeDist = std::abs(getCoordinate(node->queryEmbedding, axis)-getCoordinate(node->embedding, axis));

    if (heap.size()<K || distance(heap.top, node->queryEmbedding)> planeDist){

        if getCoordinate(node->queryEmbedding, axis) < getCoordinate(node->embedding, axis){
            knnSearch(node->right, depth+1, heap);
        }
        else{
            knnSearch(node->left, depth+1, heap);
        }
    }

    return;
}