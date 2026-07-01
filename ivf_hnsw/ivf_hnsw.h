#pragma once
#include "hnsw.h"
#include <string>
#include <vector>
#include <random>

using namespace std;

class IVFHNSW {
   public:
    IVFHNSW(int d, int nlist = 100, int nprobe = 10, int M = 16, int ef_construction = 200,
            int ef = 50, const string& space = "l2");

    void train(const vector<vector<float>>& X, int n_iter = 25);

    void add(const vector<vector<float>>& X, const vector<int>& ids = {});

    KNNResult search(const vector<float>& query, int k) const;

    int ntotal() const;
    bool is_trained() const {
        return trained_;
    }

   private:
    int d_, nlist_, nprobe_;
    bool trained_ = false;
    int next_id_ = 0;

    HNSW centroid_hnsw_;

    vector<vector<float>> centroids_; 

    vector<vector<pair<int, vector<float>>>> inverted_lists_;

    mutable mt19937 rng_{random_device{}()};

    float vec_dist(const vector<float>& a, const vector<float>& b) const;
    int nearest_centroid(const vector<float>& v) const;

    vector<vector<float>> kmeans_pp(const vector<vector<float>>& X, int k,
                                              int n_iter);
};